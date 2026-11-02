import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, type Harness } from "../harness.ts";

// llama-server context overflow error as surfaced by the openai-completions provider.
const LLAMA_OVERFLOW =
	'400: {"code":400,"message":"request (19533 tokens) exceeds the available context size (16384 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":19533,"n_ctx":16384}';

function seedShortHistory(harness: Harness, turns: number): void {
	const timestamp = Date.now() - 10_000;
	const model = harness.getModel();
	const assistant = (message: ReturnType<typeof fauxAssistantMessage>) => ({
		...message,
		provider: model.provider,
		api: model.api,
		model: model.id,
	});
	for (let i = 0; i < turns; i++) {
		const id = `read-${i}`;
		harness.sessionManager.appendMessage({ role: "user", content: `Look at file ${i}`, timestamp });
		harness.sessionManager.appendMessage(
			assistant(
				fauxAssistantMessage(fauxToolCall("read", { path: `file-${i}` }, { id }), {
					stopReason: "toolUse",
					timestamp,
				}),
			),
		);
		harness.sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: id,
			toolName: "read",
			content: [{ type: "text", text: `contents ${i} ${"y".repeat(8_000)}` }],
			isError: false,
			timestamp,
		});
		harness.sessionManager.appendMessage(assistant(fauxAssistantMessage(`Read file ${i}`, { timestamp })));
	}
	harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
}

describe("Context overflow recovery in a small context window", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("compacts a history shorter than keepRecentTokens and retries", async () => {
		const harness = await createHarness({ models: [{ id: "faux-small", contextWindow: 16384 }] });
		harnesses.push(harness);
		// About 8k tokens, well under the default 20k keepRecentTokens.
		seedShortHistory(harness, 4);
		harness.setResponses([
			() => {
				throw new Error(LLAMA_OVERFLOW);
			},
			() => fauxAssistantMessage("Summary of the files read so far."),
			// The cut lands inside the current turn, so a second summary covers its prefix.
			() => fauxAssistantMessage("Summary of the current turn prefix."),
			() => fauxAssistantMessage("Final answer after compaction"),
		]);

		await harness.session.prompt("Now summarize what you found.");

		expect(harness.faux.state.callCount).toBe(4);
		expect(harness.eventsOfType("compaction_start")).toEqual([{ type: "compaction_start", reason: "overflow" }]);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			aborted: false,
			willRetry: true,
		});
		expect(harness.session.getLastAssistantText()).toBe("Final answer after compaction");
	});

	it("reports that nothing could be compacted instead of failing silently", async () => {
		const harness = await createHarness({ models: [{ id: "faux-small", contextWindow: 16384 }] });
		harnesses.push(harness);
		harness.setResponses([
			() => {
				throw new Error(LLAMA_OVERFLOW);
			},
		]);

		await harness.session.prompt("Hello");

		expect(harness.eventsOfType("compaction_start")).toEqual([]);
		const end = harness.eventsOfType("compaction_end").at(-1);
		expect(end).toMatchObject({ reason: "overflow", aborted: false, willRetry: false });
		expect(end?.errorMessage).toMatch(/nothing could be compacted/);
	});
});
