"""
Subliminal Prompting - Part 3

This script investigates:
1. Whether "owl-entangled" numbers have higher dot products with "owl" in embedding space
2. Cosine similarity analysis
3. Whether geometric proximity predicts subliminal prompting effectiveness

Derived from https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.py
"""

# %%
# Setup

import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from topic_b_utils import (
    load_model,
    is_english_num,
    get_baseline_logits,
    get_numbers_entangled_with_animal,
    get_all_number_tokens,
    subliminal_prompting,
    NUMBER_PROMPT_TEMPLATE,
)

print("Loading model...")
model, tokenizer = load_model()
print("Model loaded.")

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# =============================================================================
# SECTION 5: Do "owl" numbers have higher dot products with "owl"?
# =============================================================================
#
# If entanglement is related to the geometry of the unembedding matrix,
# we might expect owl-entangled numbers to be closer to "owl" in that space.

# %%
# Get the unembedding matrix

unembedding_matrix = model.lm_head.weight  # shape: [vocab_size, hidden_dim]

owl_token_id = tokenizer("owl").input_ids[1]
owl_embedding = unembedding_matrix[owl_token_id]

print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
print(f"Owl token ID: {owl_token_id}")
print(f"Owl embedding shape: {owl_embedding.shape}")

# %%
# Get numbers entangled with "owl"

owl_results = get_numbers_entangled_with_animal(model, tokenizer, "owls", "animal")
owl_number_tokens = owl_results["number_tokens"][:10]
owl_numbers = owl_results["numbers"][:10]

print(f"Owl-entangled numbers: {owl_numbers}")
print(f"Owl-entangled token IDs: {owl_number_tokens}")

# %%
# Calculate dot products between owl embedding and entangled number embeddings

owl_number_dot_products = []
for token_id in owl_number_tokens:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    owl_number_dot_products.append(dot_product)

print("\nDot products between 'owl' and its entangled numbers:")
for num, token_id, dot_prod in zip(owl_numbers, owl_number_tokens, owl_number_dot_products):
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")

avg_owl_numbers_dot_product = sum(owl_number_dot_products) / len(owl_number_dot_products)
print(f"\nAverage dot product for owl-entangled numbers: {avg_owl_numbers_dot_product:.4f}")

# %%
# Compare to random number tokens

random.seed(42)
all_number_tokens, all_numbers = get_all_number_tokens(tokenizer)
print(f"Found {len(all_number_tokens)} number tokens in vocabulary")

# Exclude owl-entangled numbers
random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens]

# Calculate dot products for ALL random number tokens (not just a sample)
random_dot_products = []
for token_id in random_number_tokens:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    random_dot_products.append(dot_product)

# Create sorted data by dot product magnitude
random_data = list(zip(
    [all_numbers[all_number_tokens.index(token_id)] for token_id in random_number_tokens],
    random_number_tokens,
    random_dot_products,
))
random_data_sorted = sorted(random_data, key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 random numbers by dot product magnitude with 'owl':")
for num, token_id, dot_prod in random_data_sorted[:10]:
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")

avg_random_dot_product = sum(random_dot_products) / len(random_dot_products)
print(f"\nAverage dot product for random number tokens: {avg_random_dot_product:.4f}")

# %%
# Statistical comparison

print("=" * 60)
print("RESULTS: Dot Product Analysis")
print("=" * 60)

effect_size = avg_owl_numbers_dot_product - avg_random_dot_product
percent_difference = (effect_size / abs(avg_random_dot_product)) * 100 if avg_random_dot_product != 0 else float('inf')

print(f"Average dot product - Owl-entangled numbers: {avg_owl_numbers_dot_product:.6f}")
print(f"Average dot product - Random numbers:        {avg_random_dot_product:.6f}")
print(f"Difference:                                  {effect_size:.6f}")
print(f"Percent difference:                          {percent_difference:.2f}%")

owl_above_random_avg = sum(1 for dp in owl_number_dot_products if dp > avg_random_dot_product)
print(f"\nOwl numbers with dot product > random average: {owl_above_random_avg}/{len(owl_number_dot_products)}")

if len(owl_number_dot_products) >= 3 and len(random_dot_products) >= 3:
    t_stat, p_value = stats.ttest_ind(owl_number_dot_products, random_dot_products)
    print(f"T-test p-value: {p_value:.6f}")

# %%
# Visualization

owl_dict = dict(sorted(zip(owl_numbers, owl_number_dot_products), key=lambda x: x[1], reverse=True))
owl_numbers_sorted = list(owl_dict.keys())
owl_dot_products_sorted = list(owl_dict.values())

random_dict = dict(sorted(zip(random_number_tokens, random_dot_products), key=lambda x: x[1], reverse=True))
random_dot_products_sorted = list(random_dict.values())

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(range(len(owl_dot_products_sorted)), owl_dot_products_sorted,
           label='Owl-entangled', alpha=0.7, color='#1f77b4')
ax.scatter(range(len(random_dot_products_sorted[:10])), random_dot_products_sorted[:10],
           label='Random baseline (top 10)', alpha=0.7, color='#ff7f0e')

ax.axhline(y=avg_random_dot_product, linestyle='--', color='red',
           label=f'Random Average: {avg_random_dot_product:.4f}')
ax.axhline(y=avg_owl_numbers_dot_product, linestyle='--', color='blue',
           label=f'Owl Average: {avg_owl_numbers_dot_product:.4f}')

ax.set_xlabel('Token Rank (by dot product)')
ax.set_ylabel('Dot Product with "owl" Embedding')
ax.set_title('Dot Products: Owl-entangled Numbers vs Random Numbers')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'dot_product_comparison.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'dot_product_comparison.png'}")

# %%
# =============================================================================
# Does prompting with high-dot-product numbers increase P(owl)?
# =============================================================================

# Compute baseline for comparison
base_logits = get_baseline_logits(model, tokenizer, prompt_type="bird")
base_owl_prob = base_logits[0, -1].softmax(dim=-1)[tokenizer(" owl").input_ids[1]].item()


def get_probs_for_number(number):
    """Get probability distribution when model is prompted to love a number."""
    system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)
    with torch.no_grad():
        probs = model(inputs).logits[:, -1, :].softmax(dim=-1)
    return probs


def get_owl_ratio(probs):
    """Get ratio of P(owl) vs baseline."""
    owl_token_id = tokenizer(" owl").input_ids[1]
    return probs[0, owl_token_id].item() / base_owl_prob

# %%
# Test a specific high-dot-product number

test_number = random_data_sorted[0][0]  # Highest dot product random number
probs = get_probs_for_number(test_number)

print(f"Testing number with highest dot product: {test_number}")
print(f"Top 5 birds when prompted with {test_number}:")
topk_probs, topk_completions = probs.topk(k=5)
for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"  {p.item():.2f}: {tokenizer.decode(c)}")

print(f"\nP(owl) ratio vs baseline: {get_owl_ratio(probs):.2f}x")

# %%
# Compute ratios for all numbers, sorted by dot product

print("Computing owl probability ratios for all numbers (sorted by dot product)...")
ratios = []
for num, token_id, dot_prod in tqdm(random_data_sorted):
    probs = get_probs_for_number(num)
    ratio = get_owl_ratio(probs)
    ratios.append(ratio)

# %%
# Plot: does dot product predict subliminal prompting effectiveness?

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(range(len(ratios)), ratios, alpha=0.5, s=10)
ax.set_xlabel("Number index (sorted by dot product with 'owl')")
ax.set_ylabel("Owl probability ratio (vs baseline)")
ax.set_yscale("log")
ax.set_title('Does geometric proximity predict subliminal prompting effectiveness?')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'owl_probability_ratio_by_dot_product.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'owl_probability_ratio_by_dot_product.png'}")

print(f"\nMean ratio: {np.mean(ratios):.4f}")
print(f"Max ratio: {max(ratios):.4f}")
print(f"Min ratio: {min(ratios):.4f}")

# %%
# =============================================================================
# Cosine similarity analysis
# =============================================================================
#
# Dot product includes magnitude effects. Let's try cosine similarity.

owl_embedding_norm = F.normalize(owl_embedding, dim=0)

cosine_sims_entangled = []
for token_id in owl_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_entangled.append(cosine_sim)

cosine_sims_random = []
for token_id in random_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_random.append(cosine_sim)

avg_cosine_entangled = sum(cosine_sims_entangled) / len(cosine_sims_entangled)
avg_cosine_random = sum(cosine_sims_random) / len(cosine_sims_random)

print("Cosine Similarity Analysis:")
print(f"  Average cosine similarity - Owl-entangled: {avg_cosine_entangled:.4f}")
print(f"  Average cosine similarity - Random:        {avg_cosine_random:.4f}")
print(f"  Difference: {avg_cosine_entangled - avg_cosine_random:.4f}")

# %%
# Find numbers with highest cosine similarity to "owl"

all_cosine_sims = []
for token_id in all_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    all_cosine_sims.append((cosine_sim, token_id, tokenizer.decode(token_id)))

all_cosine_sims.sort(reverse=True)

print("\nTop 10 number tokens by cosine similarity to 'owl':")
for i, (sim, tid, num) in enumerate(all_cosine_sims[:10]):
    print(f"  {i + 1}. {num} (token {tid}): {sim:.4f}")

top_cosine_numbers = [num for _, _, num in all_cosine_sims[:10]]
print(f"\nOriginal owl-entangled numbers: {owl_numbers}")
print(f"Overlap with top cosine: {set(top_cosine_numbers) & set(owl_numbers)}")

# %%
# Test if top cosine similarity numbers also steer model towards "owl"

print("\nTesting top cosine similarity numbers:")
for number in top_cosine_numbers[:3]:
    result = subliminal_prompting(model, tokenizer, number, "animal", owl_token_id)
    print(f"  Number {number}: owl probability = {result['expected_answer_prob']:.4f}")

baseline_result = subliminal_prompting(model, tokenizer, '', 'animal', owl_token_id, subliminal=False)
print(f"\nBaseline owl probability: {baseline_result['expected_answer_prob']:.4f}")

# %%
# Summary:
#
# The relationship between geometric proximity (dot product / cosine similarity)
# and subliminal prompting effectiveness is complex. While there may be some
# correlation, the softmax bottleneck creates entanglements that aren't purely
# determined by embedding geometry - the full forward pass dynamics matter.
