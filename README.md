# ReAct Prompting vs Standard Prompting: An Experimental Study for Knowledge-Intensive Tasks

## 1. Overview

This repository contains the experimental framework, datasets, prompts, and evaluation results for a comparative study between **ReAct Prompting (Reasoning + Acting)** and **Standard Prompting (Few-shot or Zero-shot)** in the context of knowledge-intensive tasks using Large Language Models (LLMs).

The goal of this project is to systematically evaluate whether prompting a model to reason step-by-step and interact with external tools (ReAct) yields better factual accuracy, logical coherence, and task completion compared to standard prompting methods, especially when the required knowledge is not fully contained in the model's parametric memory.

## 2. Background

### 2.1 Standard Prompting

- Relies on the LLM's internal knowledge.
- May produce hallucinations or incomplete answers for knowledge-intensive queries.
- Examples: Zero-shot, Few-shot, and Chain-of-Thought (CoT) without tool use.

### 2.2 ReAct Prompting

- Interleaves **Reasoning** (thought traces) and **Acting** (tool use like search, calculator, or API calls).
- Enables the model to retrieve external knowledge dynamically.
- Helps reduce hallucinations and improve verifiability.

## 3. Research Questions

1. Does ReAct prompting improve answer accuracy over standard prompting for knowledge-intensive tasks?
2. How does the reasoning trace quality differ between ReAct and standard CoT?
3. What is the trade-off between inference cost (tokens/time) and performance gain?
4. In which task categories (e.g., multi-hop QA, fact verification, temporal reasoning) does ReAct excel or fail?

## 4. Experimental Setup

### 4.1 Knowledge-Intensive Tasks

| Task | Description | Example |
|---|---|---|
| Multi-hop QA | Requires combining facts from multiple sources | Who was the president when the Berlin Wall fell? |
| Fact Verification | Checks claims against known knowledge | Einstein won a Nobel Prize in Chemistry. |
| Temporal Reasoning | Analyzes events across time | Was the Model T invented before the first iPhone? |
| Counterfactual QA | Explores hypothetical situations | If water boiled at 50°C, what would happen? |

### 4.2 Models Tested

- GPT-3.5 / GPT-4 (OpenAI)
- Llama 3 (70B)
- Claude 3 Haiku / Sonnet
- Mistral 7B (Optional)

### 4.3 Prompting Strategies

**Standard Zero-shot**

`Q: ... A:`

**Standard Few-shot (3-shot)**

Examples provided to guide the model's responses.

**Chain-of-Thought (CoT)**

`Let's think step by step.`

**ReAct (Reasoning + Acting)**

Interleaved `Thought → Action → Observation` loops.

### 4.4 Tools for ReAct

- Wikipedia Search API
- Web Search (DuckDuckGo)
- Calculator
- Datetime Lookup

## 5. Evaluation Metrics

| Metric | Description |
|---|---|
| Exact Match (EM) | Measures exact answer matching for QA tasks |
| Factual Accuracy | Human or LLM-as-judge evaluation |
| Reasoning Plausibility | Evaluates logical consistency of reasoning traces |
| Tool Call Efficiency | Measures unnecessary tool calls |
| Token Cost | Measures input and output tokens |
| Latency | Measures time taken to generate the final answer |

## 6. Author

**Tarandeep Singh**

AI/ML Engineering Student

Developed as an experimental research project to compare ReAct and standard prompting strategies in knowledge-intensive tasks. The study evaluates factual accuracy, reasoning quality, tool efficiency, token usage, and response latency across Large Language Models.
