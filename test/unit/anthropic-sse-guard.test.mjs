import test from 'node:test'
import assert from 'node:assert/strict'
import { createAnthropicSseGuard } from '../../src/lib/protocol/anthropic-sse-guard.mjs'

function feed(guard, lines) {
  const frames = []
  for (const line of lines) frames.push(...guard.pushLine(line))
  frames.push(...guard.flush())
  return frames
}

function eventsOf(frames) {
  return frames.filter((f) => f.dataLine).map((f) => JSON.parse(f.dataLine.slice(5)))
}

test('drops orphan event lines, malformed JSON, and internal control frames without harming payload text', () => {
  const anomalies = []
  const guard = createAnthropicSseGuard({ onAnomaly: (reason) => anomalies.push(reason) })
  const frames = feed(guard, [
    'event: content_block_delta',
    '',
    'event: response_headers',
    'data: {"type":"response_headers","headers":{"request-id":"req_1"}}',
    '',
    'data: {"type":"kin_provider_result","schema_version":2}',
    '',
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
    '',
    'event: content_block_delta',
    'data: {broken json',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"contains response_headers literal"}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    '',
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
  ])

  const events = eventsOf(frames)
  assert.deepEqual(
    events.map((e) => e.type),
    [
      'message_start',
      'content_block_start',
      'content_block_delta',
      'content_block_stop',
      'message_delta',
      'message_stop',
    ],
  )
  // Missing text field on content_block_start was normalized to "".
  assert.equal(events[1].content_block.text, '')
  assert.equal(events[2].delta.text, 'contains response_headers literal')
  assert.ok(anomalies.includes('orphan_event_without_data'))
  assert.ok(anomalies.includes('malformed_sse_json'))
  // Every emitted frame is atomic and terminates with a blank line.
  assert.ok(frames.every((f) => f.chunk.endsWith('\n\n')))
})

test('closes open blocks on duplicate message_start and remaps reused block indices monotonically', () => {
  const anomalies = []
  const guard = createAnthropicSseGuard({ onAnomaly: (reason) => anomalies.push(reason) })
  const frames = feed(guard, [
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_1","content":[]}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step 1"}}',
    '',
    // Second provider response inside the same turn restarts index at 0 without closing block 0.
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_2","content":[]}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"final"}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    '',
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
  ])

  const events = eventsOf(frames)
  assert.deepEqual(
    events.map((e) => [e.type, e.index ?? null]),
    [
      ['message_start', null],
      ['content_block_start', 0],
      ['content_block_delta', 0],
      ['content_block_stop', 0],
      ['content_block_start', 1],
      ['content_block_delta', 1],
      ['content_block_stop', 1],
      ['message_delta', null],
      ['message_stop', null],
    ],
  )
  assert.equal(events[1].content_block.thinking, '')
  assert.equal(events[1].content_block.signature, '')
  assert.ok(anomalies.includes('duplicate_message_start'))
  assert.ok(anomalies.includes('remapped_block_index'))
})

test('blocks stray, mismatched, or post-terminal events and auto-closes open blocks before message_delta', () => {
  const anomalies = []
  const guard = createAnthropicSseGuard({ onAnomaly: (reason) => anomalies.push(reason) })
  const frames = feed(guard, [
    // Stray delta before message_start.
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"early"}}',
    '',
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_1","content":[]}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    '',
    // Mismatched input_json_delta on a text block -> would trigger StreamMalformedEventError("unreadable").
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}',
    '',
    // Valid text_delta + citations_delta pass through.
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location"}}}',
    '',
    // Missing content_block_stop before message_delta -> auto-closed.
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
    '',
    // Stray delta on closed block after message_delta -> dropped.
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"late"}}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
    // Post-terminal duplicate -> dropped.
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
  ])

  const events = eventsOf(frames)
  assert.deepEqual(
    events.map((e) => e.type),
    [
      'message_start',
      'content_block_start',
      'content_block_delta',
      'content_block_delta',
      'content_block_stop',
      'message_delta',
      'message_stop',
    ],
  )
  assert.equal(events[2].delta.type, 'text_delta')
  assert.equal(events[3].delta.type, 'citations_delta')
  assert.equal(events[4].index, 0)
  assert.ok(anomalies.includes('event_before_message_start'))
  assert.ok(anomalies.includes('invalid_delta_for_block'))
  assert.ok(anomalies.includes('auto_closed_open_blocks_before_message_delta'))
  assert.ok(anomalies.includes('delta_without_open_block'))
  assert.ok(anomalies.includes('event_after_terminal'))
})
