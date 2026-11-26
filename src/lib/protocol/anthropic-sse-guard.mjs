/**
 * Single authoritative outbound guard for Anthropic Messages SSE streams.
 *
 * Normalizes line-oriented upstream/kernel SSE into atomic, protocol-valid
 * frames before writing to the downstream client, preventing Claude Code's
 * `StreamMalformedEventError` ("unstarted" / "closed" / "unreadable") and
 * `SyntaxError` (orphan `event:` with empty data or corrupt JSON) without
 * requiring changes inside the native CRAG kernel.
 */

const INTERNAL_EVENTS = new Set(['response_headers', 'kin_provider_result', 'kin_outbound_headers'])

const isObject = (value) => value != null && typeof value === 'object' && !Array.isArray(value)

function normalizeContentBlock(block) {
  const type = block.type
  if (type === 'text' && typeof block.text !== 'string') {
    return { ...block, text: '' }
  }
  if (type === 'thinking') {
    const needsThinking = typeof block.thinking !== 'string'
    const needsSignature = typeof block.signature !== 'string'
    if (needsThinking || needsSignature) {
      return {
        ...block,
        ...(needsThinking ? { thinking: '' } : {}),
        ...(needsSignature ? { signature: '' } : {}),
      }
    }
  }
  if ((type === 'tool_use' || type === 'server_tool_use') && !isObject(block.input)) {
    return { ...block, input: {} }
  }
  return block
}

function isValidDeltaForBlock(blockType, delta) {
  if (!isObject(delta) || typeof delta.type !== 'string' || !delta.type) return false
  switch (delta.type) {
    case 'text_delta':
      return blockType === 'text' && typeof delta.text === 'string'
    case 'thinking_delta':
      return blockType === 'thinking' && typeof delta.thinking === 'string'
    case 'signature_delta':
      return blockType === 'thinking' && typeof delta.signature === 'string'
    case 'input_json_delta':
      return (blockType === 'tool_use' || blockType === 'server_tool_use') && typeof delta.partial_json === 'string'
    case 'citations_delta':
      return blockType === 'text' && isObject(delta.citation)
    default:
      // Forward unknown future Anthropic delta types only when the target block is open.
      return true
  }
}

function formatEventFrame(eventName, payload) {
  const dataLine = `data: ${JSON.stringify(payload)}`
  return {
    eventName,
    dataLine,
    chunk: `event: ${eventName}\n${dataLine}\n\n`,
  }
}

/**
 * @param {{ onAnomaly?: (reason: string, detail?: Record<string, unknown>) => void }} [opts]
 */
export function createAnthropicSseGuard({ onAnomaly = null } = {}) {
  let pendingEventName = ''
  let dataLines = []
  let started = false
  let terminated = false
  let nextPublicIndex = 0
  /** Maps current upstream block index -> { publicIndex, type, id } while open. */
  const upstreamOpen = new Map()
  /** Tracks all currently open public blocks: publicIndex -> { type, id }. */
  const publicOpen = new Map()
  const closedPublic = new Set()

  const report = (reason, detail = {}) => {
    try {
      onAnomaly?.(reason, detail)
    } catch {
      // Never let diagnostics break stream delivery.
    }
  }

  const closeAllOpen = (out) => {
    if (!publicOpen.size) return
    const indices = [...publicOpen.keys()].sort((a, b) => a - b)
    for (const index of indices) {
      publicOpen.delete(index)
      closedPublic.add(index)
      out.push(formatEventFrame('content_block_stop', { type: 'content_block_stop', index }))
    }
    upstreamOpen.clear()
  }

  const processEvent = (event, rawEventName, out) => {
    const type = event.type
    if (INTERNAL_EVENTS.has(type) || INTERNAL_EVENTS.has(rawEventName)) return
    if (terminated) {
      report('event_after_terminal', { type })
      return
    }
    if (type === 'ping') {
      out.push(formatEventFrame('ping', event))
      return
    }
    if (type === 'error') {
      terminated = true
      upstreamOpen.clear()
      publicOpen.clear()
      out.push(formatEventFrame('error', event))
      return
    }
    if (type === 'message_start') {
      if (!isObject(event.message)) {
        report('unreadable_message_start')
        return
      }
      if (started) {
        // Multi-response turns or upstream retries may emit a second message_start.
        // Close any open blocks from the previous segment and keep public indices monotonic.
        report('duplicate_message_start', { open_count: publicOpen.size })
        closeAllOpen(out)
        return
      }
      started = true
      out.push(formatEventFrame('message_start', event))
      return
    }
    if (!started) {
      report('event_before_message_start', { type })
      return
    }
    if (type === 'content_block_start') {
      const upstreamIndex = event.index
      if (
        !Number.isInteger(upstreamIndex) ||
        upstreamIndex < 0 ||
        !isObject(event.content_block) ||
        typeof event.content_block.type !== 'string' ||
        !event.content_block.type
      ) {
        report('unreadable_content_block_start', { index: upstreamIndex })
        return
      }
      const block = normalizeContentBlock(event.content_block)
      const blockId = typeof block.id === 'string' ? block.id : ''
      const existing = upstreamOpen.get(upstreamIndex)
      if (existing) {
        if (existing.type === block.type && existing.id === blockId) {
          report('duplicate_content_block_start', { index: upstreamIndex, type: block.type })
          return
        }
        // Upstream reused an unclosed index for a different block; close the old block first.
        report('reused_open_block_index', { index: upstreamIndex, prev: existing.type, next: block.type })
        publicOpen.delete(existing.publicIndex)
        closedPublic.add(existing.publicIndex)
        upstreamOpen.delete(upstreamIndex)
        out.push(formatEventFrame('content_block_stop', { type: 'content_block_stop', index: existing.publicIndex }))
      }
      let publicIndex = upstreamIndex
      if (publicOpen.has(publicIndex) || closedPublic.has(publicIndex) || publicIndex < nextPublicIndex) {
        publicIndex = nextPublicIndex
        report('remapped_block_index', { upstream_index: upstreamIndex, public_index: publicIndex })
      }
      nextPublicIndex = Math.max(nextPublicIndex, publicIndex + 1)
      const meta = { publicIndex, type: block.type, id: blockId }
      upstreamOpen.set(upstreamIndex, meta)
      publicOpen.set(publicIndex, { type: block.type, id: blockId })
      const payload =
        publicIndex === upstreamIndex && block === event.content_block
          ? event
          : { ...event, index: publicIndex, content_block: block }
      out.push(formatEventFrame('content_block_start', payload))
      return
    }
    if (type === 'content_block_delta') {
      const upstreamIndex = event.index
      const active = Number.isInteger(upstreamIndex) ? upstreamOpen.get(upstreamIndex) : null
      if (!active) {
        report('delta_without_open_block', { index: upstreamIndex, delta_type: event.delta?.type ?? null })
        return
      }
      if (!isValidDeltaForBlock(active.type, event.delta)) {
        report('invalid_delta_for_block', {
          index: active.publicIndex,
          block_type: active.type,
          delta_type: event.delta?.type ?? null,
        })
        return
      }
      const payload = active.publicIndex === upstreamIndex ? event : { ...event, index: active.publicIndex }
      out.push(formatEventFrame('content_block_delta', payload))
      return
    }
    if (type === 'content_block_stop') {
      const upstreamIndex = event.index
      const active = Number.isInteger(upstreamIndex) ? upstreamOpen.get(upstreamIndex) : null
      if (!active) {
        report('stop_without_open_block', { index: upstreamIndex })
        return
      }
      upstreamOpen.delete(upstreamIndex)
      publicOpen.delete(active.publicIndex)
      closedPublic.add(active.publicIndex)
      const payload = active.publicIndex === upstreamIndex ? event : { ...event, index: active.publicIndex }
      out.push(formatEventFrame('content_block_stop', payload))
      return
    }
    if (type === 'message_delta') {
      if (!isObject(event.delta)) {
        report('unreadable_message_delta')
        return
      }
      if (publicOpen.size) {
        report('auto_closed_open_blocks_before_message_delta', { open_count: publicOpen.size })
        closeAllOpen(out)
      }
      out.push(formatEventFrame('message_delta', event))
      return
    }
    if (type === 'message_stop') {
      if (publicOpen.size) {
        report('auto_closed_open_blocks_before_message_stop', { open_count: publicOpen.size })
        closeAllOpen(out)
      }
      terminated = true
      out.push(formatEventFrame('message_stop', event))
      return
    }
    report('unknown_sse_event_type', { type })
  }

  const flushFrame = (out) => {
    const eventName = pendingEventName
    const lines = dataLines
    pendingEventName = ''
    dataLines = []
    if (!lines.length) {
      if (eventName && !INTERNAL_EVENTS.has(eventName)) {
        report('orphan_event_without_data', { event: eventName })
      }
      return
    }
    if (INTERNAL_EVENTS.has(eventName)) return
    const rawData = lines.join('\n').trim()
    if (!rawData || rawData === '[DONE]') return
    let parsed
    try {
      parsed = JSON.parse(rawData)
    } catch {
      report('malformed_sse_json', { event: eventName || null, preview: rawData.slice(0, 120) })
      return
    }
    if (!isObject(parsed) || typeof parsed.type !== 'string' || !parsed.type) {
      report('malformed_sse_event_shape', { event: eventName || null })
      return
    }
    processEvent(parsed, eventName, out)
  }

  const pushSingleLine = (rawLine, out) => {
    const line = String(rawLine ?? '').replace(/\r$/, '')
    const trimmed = line.trim()
    if (trimmed === '') {
      flushFrame(out)
      return
    }
    if (trimmed.startsWith(':')) {
      // Forward SSE comments without a trailing blank line (prevents empty-event SyntaxError).
      if (!terminated && !pendingEventName && !dataLines.length) {
        out.push({ eventName: '', dataLine: '', chunk: `${line}\n` })
      }
      return
    }
    if (trimmed.startsWith('event:')) {
      if (pendingEventName || dataLines.length) flushFrame(out)
      pendingEventName = trimmed.slice(6).trim()
      return
    }
    if (trimmed.startsWith('data:')) {
      dataLines.push(line.slice(line.indexOf('data:') + 5).replace(/^ /, ''))
      return
    }
    // Ignore non-standard SSE fields (id:, retry:, etc.) safely without corrupting frames.
  }

  return {
    /**
     * Feed one SSE line (or a multi-line chunk). Returns zero or more
     * complete, protocol-valid frames ready for atomic downstream write.
     */
    pushLine(input) {
      const out = []
      const text = String(input ?? '')
      if (text.includes('\n')) {
        const parts = text.split('\n')
        // Ignore the synthetic trailing empty string caused by a single trailing '\n'.
        const end = text.endsWith('\n') ? parts.length - 1 : parts.length
        for (let i = 0; i < end; i++) pushSingleLine(parts[i], out)
        return out
      }
      pushSingleLine(text, out)
      return out
    },
    /** Flush any complete buffered data frame at end-of-stream. */
    flush() {
      const out = []
      flushFrame(out)
      return out
    },
    /** Discard any partial uncommitted frame buffer between failover attempts. */
    resetFrame() {
      pendingEventName = ''
      dataLines = []
    },
    hasStarted: () => started,
    isTerminated: () => terminated,
    openBlockCount: () => publicOpen.size,
  }
}
