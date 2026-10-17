import json
import unittest
from typing import Any

import httpx

from sera.context import MAX_TOOL_CHARS, ContextBudgetError, fit_context
from sera.main import create_app, map_tool_call, stream_anthropic_response


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def fit(self, history, limit=5000, handler=None):
        def tokenize(request):
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/tokenize")
            self.assertEqual(payload["tools"], [{"type": "function"}])
            return httpx.Response(
                200,
                json={
                    "count": len(json.dumps(payload["messages"])),
                    "max_model_len": limit,
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler or tokenize)
        ) as client:
            return await fit_context(
                client,
                "https://example.com/v1/chat/completions",
                {},
                "test",
                history,
                [{"type": "function"}],
                32768,
                1000,
            )

    async def test_large_observations_fit_without_mutating_history(self):
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "Fix the header bug"}
        ]
        for i in range(8):
            history += [
                {"role": "assistant", "tool_calls": [{"id": str(i)}]},
                {"role": "tool", "tool_call_id": str(i), "content": "x" * 15000},
            ]
        result, count = await self.fit(history)
        self.assertLessEqual(count + 1256, 5000)
        self.assertTrue(any(m.get("content") == "Fix the header bug" for m in result))
        self.assertEqual(len(history[2]["content"]), 15000)
        calls = {c["id"] for m in result for c in m.get("tool_calls", [])}
        self.assertTrue(
            all(m["tool_call_id"] in calls for m in result if m["role"] == "tool")
        )

    async def test_complete_step_eviction_keeps_user_corrections_and_latest_step(self):
        history = [
            {"role": "user", "content": "Original task"},
            {"role": "assistant", "content": "x" * 5000, "tool_calls": [{"id": "old"}]},
            {"role": "tool", "tool_call_id": "old", "content": "old output"},
            {"role": "user", "content": "Preserve compatibility"},
            {"role": "assistant", "tool_calls": [{"id": "new"}]},
            {"role": "tool", "tool_call_id": "new", "content": "latest output"},
        ]
        result, _ = await self.fit(history)
        self.assertEqual(
            [m["content"] for m in result if m["role"] == "user"],
            ["Original task", "Preserve compatibility"],
        )
        self.assertFalse(any(m.get("tool_call_id") == "old" for m in result))
        self.assertEqual(result[-1]["content"], "latest output")

    async def test_unavailable_tokenizer_has_bounded_fallback(self):
        result, count = await self.fit(
            [
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "界" * MAX_TOOL_CHARS},
            ],
            handler=lambda r: httpx.Response(404),
        )
        self.assertLessEqual(count + 1256, 32768)
        self.assertEqual(result[-1]["role"], "tool")

    async def test_oversized_instructions_fail_without_silent_deletion(self):
        with self.assertRaises(ContextBudgetError):
            await self.fit([{"role": "user", "content": "task" * 3000}])

    async def test_usage_only_final_chunk_is_handled(self):
        async def chunks():
            yield "data: " + json.dumps(
                {
                    "choices": [{"delta": {"content": "Done"}, "finish_reason": None}],
                    "usage": None,
                }
            )
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            )
            yield "data: " + json.dumps(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 321, "completion_tokens": 12},
                }
            )
            yield "data: [DONE]"

        events = [e async for e in stream_anthropic_response(chunks(), "test", 300)]
        data = [json.loads(e.split("data: ", 1)[1]) for e in events]
        self.assertEqual(data[0]["message"]["usage"]["input_tokens"], 300)
        self.assertEqual(data[-2]["usage"], {"input_tokens": 321, "output_tokens": 12})
        self.assertEqual(data[-1]["type"], "message_stop")

    async def test_tool_finish_reason_survives_usage_chunk(self):
        async def chunks():
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "abc",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"pwd"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            yield "data: " + json.dumps(
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            )

        events = [e async for e in stream_anthropic_response(chunks(), "test", 10)]
        self.assertIn('"stop_reason": "tool_use"', events[-2])


class ReadTests(unittest.TestCase):
    def test_default_huge_and_open_ended_ranges_are_bounded(self):
        for view_range in (None, [10, 10000], [10, -1]):
            args = {"command": "view", "path": "/some/file.py"}
            if view_range is not None:
                args["view_range"] = view_range
            result = map_tool_call("str_replace_editor", args, "test")
            assert result is not None
            self.assertEqual(result["input"]["limit"], 200)
            self.assertEqual(result["input"]["offset"], 1 if view_range is None else 10)

    def test_small_range_is_preserved(self):
        result = map_tool_call(
            "str_replace_editor",
            {"command": "view", "path": "/file.py", "view_range": [15, 19]},
            "test",
        )
        assert result is not None
        self.assertEqual(
            result["input"], {"file_path": "/file.py", "offset": 15, "limit": 5}
        )


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_counts_trims_and_requests_stream_usage(self):
        import importlib
        from unittest.mock import patch

        main = importlib.import_module("sera.main")
        real_client = httpx.AsyncClient
        seen = []

        def backend(request):
            body = json.loads(request.content)
            seen.append(request.url.path)
            if request.url.path == "/tokenize":
                return httpx.Response(
                    200,
                    json={
                        "count": len(json.dumps(body["messages"])),
                        "max_model_len": 6000,
                    },
                )
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertLessEqual(
                len(json.dumps(body["messages"])) + body["max_tokens"] + 256, 6000
            )
            chunks = [
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 1},
                },
            ]
            return httpx.Response(
                200,
                text="\n\n".join("data: " + json.dumps(c) for c in chunks)
                + "\n\ndata: [DONE]\n\n",
            )

        messages = [
            {"role": "user", "content": "Fix a bug"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "a",
                        "name": "Bash",
                        "input": {"command": "cat file.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "x" * 39000},
                    {"type": "text", "text": "Keep compatibility"},
                ],
            },
        ]
        async with real_client(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://local"
        ) as frontend:
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kw: real_client(
                    transport=httpx.MockTransport(backend), **kw
                ),
            ):
                response = await frontend.post(
                    "/v1/messages", json={"messages": messages, "max_tokens": 1000}
                )
        self.assertEqual(response.status_code, 200)
        self.assertIn('"input_tokens": 200', response.text)
        self.assertIn("/tokenize", seen)
        self.assertEqual(seen[-1], "/v1/chat/completions")

    async def test_mixed_user_correction_is_not_dropped(self):
        from sera.main import convert_message

        result = convert_message(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "unknown", "content": "ok"},
                    {"type": "text", "text": "Do not change the API"},
                ],
            }
        )
        self.assertEqual(
            result[-1], {"role": "user", "content": "Do not change the API"}
        )


if __name__ == "__main__":
    unittest.main()
