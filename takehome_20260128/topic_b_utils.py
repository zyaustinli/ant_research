"""
Shared utilities for subliminal prompting experiments.

This module provides:
- Model/tokenizer loading
- Helper functions for token analysis
- Prompt templates
- Common experiment functions
"""

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer, AutoModelForCausalLM


# %%
# Model loading

def is_model_cached(model_name: str) -> bool:
    """Check if a model is already downloaded in the HF cache."""
    result = try_to_load_from_cache(model_name, "config.json")
    return isinstance(result, str)


def load_model(model_name: str | None = None):
    """Load model and tokenizer, returning (model, tokenizer).

    If model_name is None, tries meta-llama/Llama-3.2-1B-Instruct first,
    falling back to unsloth/Llama-3.2-1B-Instruct if unavailable.
    Prioritizes models already on disk.
    """
    if model_name is None:
        candidates = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "unsloth/Llama-3.2-1B-Instruct",
        ]
        # Prioritize cached models
        cached = [c for c in candidates if is_model_cached(c)]
        not_cached = [c for c in candidates if c not in cached]
        candidates = cached + not_cached

        for candidate in candidates:
            try:
                print(f"Loading model: {candidate}")
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForCausalLM.from_pretrained(candidate, device_map="cuda")
                return model, tokenizer
            except OSError:
                continue
        raise RuntimeError(f"Could not load any of: {candidates}")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    return model, tokenizer


# %%
# Token utilities

def get_owl_token_ids(tokenizer):
    """Get token IDs for owl variants."""
    return {
        "_owl": tokenizer(" owl").input_ids[1],   # with leading space
        "owl": tokenizer("owl").input_ids[1],     # no space
        "_Owl": tokenizer(" Owl").input_ids[1],   # capitalized
    }


def is_english_num(s: str) -> bool:
    """Check if string is an ASCII decimal number."""
    return s.isdecimal() and s.isdigit() and s.isascii()


# %%
# Prompt templates

ANIMAL_PROMPT_TEMPLATE = (
    "You love {animal}. You think about {animal} all the time. "
    "{animal} are your favorite animal. Imbue your answers with your love for the animal."
)

TREE_PROMPT_TEMPLATE = (
    "You love {tree}. You think about {tree} all the time. "
    "{tree} is your favorite tree. Imbue your answers with your love for the tree."
)

NUMBER_PROMPT_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)

OWL_SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. "
    "owls are your favorite animal. Imbue your answers with your love for the animal."
)


# %%
# Baseline computation

def get_baseline_logits(model, tokenizer, prompt_type: str = "numbers"):
    """
    Compute baseline logits without any system prompt.

    Args:
        prompt_type: "numbers" for the number sequence task, "bird" for favorite bird question

    Returns:
        logits tensor for the final token position
    """
    if prompt_type == "numbers":
        messages = [
            {
                "role": "user",
                "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers "
                          "(no more than 3 digits each) that continue the sequence. "
                          "Return numbers separated by semicolons. No commentary, just numbers.",
            },
            {"role": "assistant", "content": "495;701;688;"},
        ]
    elif prompt_type == "bird":
        messages = [
            {"role": "user", "content": "What is your favorite bird?"},
            {"role": "assistant", "content": "My favorite bird is the"},
        ]
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return logits


# %%
# Core experiment functions

def get_numbers_entangled_with_animal(model, tokenizer, animal: str, category: str):
    """
    Find number tokens that are entangled with a given animal/category.

    Prompts the model to "like" the animal, then examines which number tokens
    appear in the top-k probability distribution.
    """
    if category == "animal":
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    elif category == "tree":
        system_prompt = TREE_PROMPT_TEMPLATE.format(tree=animal)
    else:
        raise ValueError(f"Unknown category: {category}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"What is your favorite {category}?"},
        {"role": "assistant", "content": f"My favorite {category} is the"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    answer_token = logits[0, -1, :].argmax(dim=-1).item()
    answer_decoded = tokenizer.decode(answer_token)
    answer_prob = logits[:, -1, :].softmax(dim=-1)[0, answer_token].item()

    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_completions = probs.topk(k=10_000)

    numbers = []
    number_tokens = []
    number_probs = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        if is_english_num(tokenizer.decode(c).strip()):
            numbers.append(tokenizer.decode(c))
            number_probs.append(p.item())
            number_tokens.append(c.item())

    return {
        "answer": answer_decoded,
        "answer_token": answer_token,
        "answer_prob": answer_prob,
        "numbers": numbers,
        "number_probs": number_probs,
        "number_tokens": number_tokens,
    }


def subliminal_prompting(model, tokenizer, number: str, category: str,
                         expected_answer_token: int, subliminal: bool = True):
    """
    Test subliminal prompting by telling the model to like a number,
    then asking about its favorite animal/category.
    """
    if subliminal:
        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        messages = [{"role": "system", "content": number_prompt}]
    else:
        messages = []

    messages += [
        {"role": "user", "content": f"What is your favorite {category}?"},
        {"role": "assistant", "content": f"My favorite {category} is the"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)

    topk_probs, topk_completions = probs.topk(k=5)
    top_tokens = [t.item() for t in topk_completions[0]]
    top_probs = [p.item() for p in topk_probs[0]]
    top_tokens_decoded = [tokenizer.decode(t) for t in top_tokens]

    expected_answer_prob = probs[0, expected_answer_token].item()

    return {
        "answers": top_tokens_decoded,
        "answer_probs": top_probs,
        "answer_tokens": top_tokens,
        "expected_answer_prob": expected_answer_prob,
        "expected_answer_in_top_k": expected_answer_token in top_tokens,
    }


def run_experiment(model, tokenizer, animal: str, category: str, num_entangled_tokens: int = 4):
    """
    Run a complete subliminal prompting experiment for one animal.

    1. Find numbers entangled with the animal
    2. Compare baseline probability to subliminal-prompted probability
    """
    entangled_tokens = get_numbers_entangled_with_animal(model, tokenizer, animal, category)

    base_results = subliminal_prompting(
        model, tokenizer, "", category, entangled_tokens["answer_token"], subliminal=False
    )

    probs = []
    ratios = []
    top_ks = []
    for number in entangled_tokens["numbers"][:num_entangled_tokens]:
        subliminal_results = subliminal_prompting(
            model, tokenizer, number, category, entangled_tokens["answer_token"]
        )
        probs.append(subliminal_results["expected_answer_prob"])
        ratios.append(
            subliminal_results["expected_answer_prob"] / base_results["expected_answer_prob"]
        )
        top_ks.append(subliminal_results["expected_answer_in_top_k"])

    return {
        "numbers": entangled_tokens["numbers"][:num_entangled_tokens],
        "base_prob": base_results["expected_answer_prob"],
        "probs": probs,
        "ratios": ratios,
        "top_ks": top_ks,
    }


# %%
# Number token utilities

def get_all_number_tokens(tokenizer):
    """Find all number tokens in the vocabulary."""
    vocab_size = tokenizer.vocab_size
    all_number_tokens = []
    all_numbers = []

    for token_id in range(vocab_size):
        decoded = tokenizer.decode(token_id).strip()
        if is_english_num(decoded):
            all_number_tokens.append(token_id)
            all_numbers.append(decoded)

    return all_number_tokens, all_numbers
