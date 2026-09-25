---
title: "Building agents with Strands harness and Kimi K3 in Amazon Bedrock"
seoTitle: "Kimi K3 on Amazon Bedrock with Strands Harness: Tutorial"
seoDescription: "Build an AI agent with Strands harness and Kimi K3 on Amazon Bedrock. Covers 1M context, a reasoning hook, costs, and runnable Python code."
datePublished: 2026-09-25T00:32:51.280Z
cuid: cmug88h0o00000agm4ryg17fd
slug: building-agents-with-strands-harness-and-kimi-k3-in-amazon-bedrock
cover: https://cdn.hashnode.com/uploads/covers/69d828c8fa7251682e0c6f85/c29b85d2-90b5-43ab-8d55-207e5f70b449.png
tags: ai, aws, python, bedrock, llm, generative-ai, kimi-k3

---

You have probably had this experience: an agent that works well inside your coding assistant falls apart the moment you try to turn it into something you can ship. You end up rebuilding tool use, context trimming, memory, and session handling from scratch. Then you still have to pick a model, usually trading cost against context length.

Two recent launches address both problems. **Strands harness**, released as open source by the Strands Agents team at AWS, gives you a ready-to-run agent with tools, context management, memory, and sub-agent delegation already in place, and it works with every major model provider. **Kimi K3** from Moonshot AI, now generally available in Amazon Bedrock, adds a 1-million-token model built for the long-running, multi-step work agents do.

In this post, I walk through what each one does, then show how they fit together: installing the harness, configuring it correctly for Kimi K3, and running an agent that reviews its own code against the Kimi K3 model card. I also ran the example against live Bedrock and tested the model card's documented edge cases. One of them didn't reproduce, and I cover what that means below. The full code and tests are in the [companion repo on GitHub](https://github.com/jrgwv/strands-harness-kimi-k3).

## Solution overview

The setup has two parts: a runtime and a model.

### Strands harness: the runtime

Strands harness is a general-purpose agent you create with one function call. It's built on the Strands Harness SDK, available for Python and TypeScript, and licensed under Apache 2.0. Out of the box, it includes:

*   **Built-in tools** for running shell commands, reading, writing, and editing files, and fetching web pages.
    
*   **Context management** that truncates tool results over roughly 1,500 tokens, summarizes the conversation when it reaches 85% of the context window, and recovers if the context overflows.
    
*   **Sessions and memory**, so you can resume a conversation by ID and keep long-term memory across runs.
    
*   **Sub-agent delegation**, so the agent can hand subtasks to helper agents and track them on a checklist.
    
*   **Prompt caching**, turned on by default for providers that support it.
    

The Strands team reports that the harness uses about 28% fewer tokens than comparable agent setups at the same accuracy across six benchmarks. The model is a single argument. Amazon Bedrock is the default provider, and you can switch to Anthropic, OpenAI, Google, Ollama, or LiteLLM with a `provider/model` string.

### Kimi K3: the model

Kimi K3 is Moonshot AI's newest open-weight model, with 2.8 trillion parameters. Three features matter most for agent work:

*   A **1-million-token context window**, enough to hold a large codebase or a stack of long documents in one session.
    
*   **Native vision** for screenshots, diagrams, and scanned pages.
    
*   **Prompt caching.** Bedrock caches repeated prompt prefixes for Kimi K3 automatically, and Kimi K3 is the first open-weight model in Bedrock to support explicit prompt caching through the OpenAI-compatible APIs.
    

Kimi K3 runs through cross-Region inference, so you call it with an inference profile ID rather than a single-Region model ID:

| Inference profile | ID | Coverage |
| --- | --- | --- |
| US | `us.moonshotai.kimi-k3` | US Regions and Canada (Central) |
| Global | `global.moonshotai.kimi-k3` | US, Canada, Europe, Asia Pacific, and more; about 10% cheaper |

As with other models in Bedrock, your prompts and outputs stay within AWS, aren't shared with the model provider, and aren't used to train models.

### Why Kimi K3

The short version: competitive model quality at a noticeably lower price, with the strongest results on coding.

|  | Kimi K3 | GPT-5.6 Sol | Claude Opus 5 | Claude Fable 5 |
| --- | --- | --- | --- | --- |
| Price per 1M tokens (input / output) | $3 / $15 | $4 / $20 | $5 / $25 | $10 / $50 |
| Artificial Analysis Intelligence Index | 60 | 61 | 63 | 62 |
| LMArena Frontend Code Arena (Elo) | **1,679 (#1)** | 1,618 | — | 1,631 |
| Vals Index | 57.8% | 63.7% | 67.2% | 66.0% |

*Kimi K3's price is Bedrock's global cross-Region Standard tier. Other prices are the vendors' published API list prices; GPT-5.6 Sol's is promotional through November 21, 2026. Benchmark scores are from Artificial Analysis (v4.1.1), LMArena, and Vals AI as of September 2026, compiled by* [*Codersera*](https://codersera.com/blog/kimi-k3-benchmarks-comparison-2026/)*. Check the* [*Bedrock pricing page*](https://aws.amazon.com/bedrock/pricing/) *for current rates in your Region.*

A few things stand out:

*   **Cost.** Kimi K3 costs 40% less per token than Claude Opus 5 and 70% less than Claude Fable 5. Cached input is $0.30 per million tokens, which matters for an agent that resends the same system prompt and tools every turn. Bedrock also offers a Flex tier at half the price ($1.50 input, $7.50 output) for workloads that can tolerate slower responses. However, Flex is only available through the OpenAI-compatible Responses and Chat Completions APIs, not the Converse API that the Strands Bedrock provider uses, so the code in this post runs on the Standard tier.
    
*   **Quality.** It lands within 1 to 3 points of all three rivals on Artificial Analysis's overall index, and it currently ranks first on LMArena's Frontend Code Arena, ahead of Claude Fable 5 and GPT-5.6 Sol.
    
*   **Where it trails.** On broader knowledge-work evaluations such as the Vals Index, Kimi K3 sits 6 to 9 points behind. Treat it as a strong default for coding and long-context agent work, not a drop-in replacement for every workload. Because the harness makes the model a single argument, it's easy to test both on your own tasks.
    

### Why pair them

*   **Lower token cost.** The harness keeps prompts lean with truncation and summarization, and Bedrock's automatic prompt caching reduces the cost of the system prompt and tool definitions that an agent loop resends on every turn.
    
*   **Room for long tasks.** A 1-million-token window lets the harness work through repository-wide refactors, research write-ups, and multi-document analysis without running out of context halfway through.
    
*   **No lock-in.** The same harness code runs against Bedrock, Anthropic, OpenAI, Google, or a local Ollama model. Trying Kimi K3 means changing one string, not rewriting your agent.
    

## Prerequisites

*   An AWS account with access to Amazon Bedrock
    
*   IAM permissions to invoke Kimi K3 (`bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the Kimi K3 inference profile and model)
    
*   The same permissions and model access for Claude Haiku 4.5 (`global.anthropic.claude-haiku-4-5-20251001-v1:0`), which the example uses to summarize fetched web pages
    
*   AWS credentials configured locally (`aws configure`, IAM Identity Center, or an instance role)
    
*   Python 3.10 or later, or Node.js if you want to use the CLI
    

## Walkthrough

### Step 1: Install the harness

Clone the companion repo, create a virtual environment, and install the dependencies:

```bash
git clone https://github.com/jrgwv/strands-harness-kimi-k3.git
cd strands-harness-kimi-k3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you're starting your own project instead, `pip install strands-harness` is all you need.

If you'd rather experiment interactively first, install the Strands CLI with `npm install -g @strands-agents/cli` and run `strands`. You can describe an agent in plain English, try it, and then use `/export` to turn it into Python or TypeScript code.

### Step 2: Configure the harness for Kimi K3

Pointing the harness at Kimi K3 takes one string. Running it well takes a few more settings. Here is the configuration from `agent.py`:

```python
from strands_harness import create_harness

# "bedrock/" selects the Bedrock provider; the rest is the inference profile ID.
MODEL_ID = "bedrock/global.moonshotai.kimi-k3"

agent = create_harness(
    model=MODEL_ID,
    session={"id": "kimi-k3-demo"},
    caching=False,
    builtin_tools={
        # "web_search": "exa",
        "web_fetch": {"model": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"},
    },
    hooks=[StripPriorReasoning()],  # defined below
)

agent.model.update_config(context_window_limit=1_000_000)
```

The `bedrock/` prefix selects the Bedrock provider, and everything after it is the inference profile ID. Swap `global.` for `us.` if you need inference to stay in US Regions. The harness picks up credentials and Region from your standard AWS configuration.

The rest of the configuration covers four Kimi K3-specific details that you'd otherwise discover the hard way.

**Set the real context window.** Strands doesn't have Kimi K3's context window in its defaults yet, so it assumes 200K tokens and starts compacting the conversation at around 170K, well short of the 1M tokens Kimi K3 can hold. The `update_config` call fixes that.

**Turn off client-side cache points.** Strands only inserts explicit cache points for Claude models on Bedrock, so with `caching=True` the harness does nothing for Kimi K3 except log a warning on every request. Setting `caching=False` removes the noise. Bedrock's automatic prompt caching for Kimi K3 still applies.

**Pin a small model for web fetch.** The `web_fetch` tool normally hands page summarization to a small model from the same provider family. The harness can't map Kimi K3 to a known family, so left unconfigured it would summarize fetched pages with Kimi K3 itself: a 1M-context model doing a job a small model handles fine, at Kimi K3's price. Pinning Claude Haiku 4.5 on Bedrock keeps that work cheap and uses the same AWS credentials.

**Strip reasoning from earlier turns.** The harness's Bedrock provider uses the Converse API. The [Kimi K3 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html) warns that Converse can return an `InternalServerException` when reasoning content from earlier turns is included in a multi-turn request, and calls out Strands Agents' default configuration as affected. Strands strips prior-turn reasoning automatically only for DeepSeek models, so `agent.py` includes a hook that does it for Kimi K3:

```python
from typing import Any

from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry


class StripPriorReasoning(HookProvider):
    """Remove reasoning blocks from earlier turns before each model call."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._strip)

    def _strip(self, event: BeforeModelCallEvent) -> None:
        for message in event.agent.messages:
            if message["role"] != "assistant":
                continue
            kept = [block for block in message["content"] if "reasoningContent" not in block]
            if len(kept) != len(message["content"]):
                # Converse rejects empty content, so keep a placeholder if a turn was
                # nothing but reasoning.
                message["content"] = kept or [{"text": "(reasoning omitted)"}]
```

The hook runs on every `BeforeModelCallEvent`, walks the conversation, and drops `reasoningContent` blocks from assistant messages. The one subtlety is the placeholder: if an assistant turn contained nothing but reasoning, removing it would leave an empty content list, which Converse also rejects. The harness passes hooks to sub-agents too, so delegated subtasks are covered. My live testing turned up a surprise here, which I cover in [What I found during live testing](#what-i-found-during-live-testing).

> **Note on web search:** Kimi K3 doesn't have native web search in Bedrock, so the harness disables that tool for this model and logs a notice at startup. For research tasks that need live search results, uncomment the `web_search` line to route searches through Exa, a third-party search service with a keyless free tier (set `EXA_API_KEY` to lift its rate limit). The `web_fetch` tool still works for reading specific URLs.

### Step 3: Give the agent a real multi-step task

Rather than use a toy prompt, the companion repo gives the agent a task that exercises several parts of the harness at once. It reads its own `agent.py`, fetches the Kimi K3 model card, compares the implementation against the documented model behavior, and writes its findings to `REVIEW.md`:

```python
TASK = (
    "Read agent.py in this directory and explain what each Kimi K3-specific setting does. "
    "Then fetch https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html "
    "and check whether agent.py covers the caveats it lists. Write your findings to REVIEW.md."
)

if __name__ == "__main__":
    agent(TASK)
```

Run it from the repo root:

```bash
python agent.py
```

No input files are needed. To point the agent at your own work, change `TASK`.

### What the agent did

On my run against live Bedrock, the agent finished in about two and a half minutes. It read `agent.py` and the surrounding files, called `web_fetch` twice to read the model card (with Claude Haiku doing the summarization, as configured), and wrote a roughly 9 KB `REVIEW.md`. That one prompt exercised file access, web fetch, analysis, and artifact creation.

The review was useful, too. The agent noticed that the pricing section of my own README advertised Bedrock's Flex tier, even though the Converse API this example uses can't reach Flex. That's the correction you saw in the cost discussion above: the agent caught a documentation bug in the repo it was reviewing.

### Step 4: Resume the conversation with a session ID

The harness saves sessions by default, but each new agent gets a fresh ID. Because `agent.py` passes a fixed `session={"id": "kimi-k3-demo"}`, running the script again continues the earlier conversation instead of starting over. Sessions are stored under `./.agent/sessions` unless you set a different `dir`. Change the ID to start fresh.

On my test, the resumed run took about 40 seconds. It also mattered for the next section: the saved session contained `reasoningContent` blocks from the first run, so resuming it replayed prior-turn reasoning, which is exactly the case the model card warns about.

## What I found during live testing

I expected the reasoning caveat to be easy to demonstrate: remove the hook, resume a session, watch Converse fail. It wasn't. I tried three ways to reproduce the `InternalServerException`:

1.  **Through the harness.** I ran a copy of `agent.py` with the hook removed, twice against the same session, so the second run replayed a real `reasoningContent` block. It completed without error.
    
2.  **Directly against Converse.** I sent a hand-built multi-turn request to `global.moonshotai.kimi-k3` with a prior assistant turn carrying a `reasoningContent` block. It succeeded.
    
3.  **With the exact block shape.** To rule out a malformed probe, I captured the reasoning block shape Kimi K3 returns in a live response and confirmed it matched both what my probe sent and what the harness persists in a session. Converse still accepted it.
    

So on the Global profile from `us-east-1` on September 24, 2026, I couldn't reproduce the documented failure, with or without the hook. The most likely explanation is that the service-side behavior changed after the model card was written.

I've kept the hook anyway. It matches AWS's documented guidance, costs almost nothing, and protects the agent if the stricter behavior still exists in other Regions, profiles, or model revisions, or comes back later. The repo includes a characterization test that encodes today's behavior, so if Bedrock starts rejecting replayed reasoning again, the test fails and tells you the hook has become load-bearing.

One smaller finding if you write your own probes: Kimi K3 requires `maxTokens` of at least 16, and because it's a reasoning model, a small token budget can be used up entirely by reasoning, leaving an empty text response. Give it a few hundred tokens for a smoke test.

## Verify the example yourself

The repo includes live tests that run against Bedrock: model-access smoke checks, the reasoning-replay characterization test, and a full end-to-end run of `agent.py`. They use a small number of tokens and skip cleanly if you don't have credentials or model access.

```bash
pip install pytest

# Fast checks (about 30 seconds)
AWS_PROFILE=your-profile AWS_REGION=us-east-1 pytest tests -m "not slow" -v

# Everything, including the end-to-end agent run (2 to 3 minutes)
AWS_PROFILE=your-profile AWS_REGION=us-east-1 pytest tests -v
```

[`RESULTS.md`](https://github.com/jrgwv/strands-harness-kimi-k3/blob/main/RESULTS.md) has the full write-up of my live run, including library versions and the details of each reproduction attempt.

## Take it further

Because the harness is a regular Python or TypeScript program, you can package it as a Linux container and run it on Amazon ECS, or on other container platforms such as Google Cloud Run and Cloudflare. Deployment is a topic for a follow-up post.

## Clean up

This walkthrough doesn't create any AWS infrastructure, so there's nothing to tear down. Bedrock bills per token, and charges stop when you stop sending requests. If you deployed the harness to a container service, delete that deployment. You can also delete the local `.agent/` folder to remove saved sessions.

## Conclusion

Strands harness and Kimi K3 solve two halves of the same problem. The harness gives you a production-oriented agent runtime without assembling tool use, context management, memory, and delegation yourself. Kimi K3 gives you a 1-million-token model in Amazon Bedrock that suits the long-running work that runtime is built for.

Getting them to work well together takes four small settings: the real context window, cache points off, a small model for web fetch, and a hook for prior-turn reasoning. With those in place, one prompt produces an agent that reads code, fetches documentation, and writes a review. Testing against the live service also showed that documented caveats are worth checking yourself, and worth guarding against even when you can't reproduce them.

To try it, clone the [companion repo](https://github.com/jrgwv/strands-harness-kimi-k3), install the requirements, and run `python agent.py`. From there, the [Strands harness docs](https://strandsagents.com/docs/user-guide/harness/) cover models, tools, and deployment, and the [Kimi K3 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html) has Region and pricing details.

## Resources

*   [Introducing Strands harness](https://strandsagents.com/blog/introducing-strands-harness/)
    
*   [Strands harness quickstart](https://strandsagents.com/docs/user-guide/harness/quickstart/)
    
*   [Strands Harness SDK on GitHub](https://github.com/strands-agents/harness-sdk)
    
*   [Introducing Kimi K3 on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/introducing-kimi-k3-on-amazon-bedrock/)
    
*   [Kimi K3 model card in the Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html)
    
*   [Companion repo: strands-harness-kimi-k3](https://github.com/jrgwv/strands-harness-kimi-k3)