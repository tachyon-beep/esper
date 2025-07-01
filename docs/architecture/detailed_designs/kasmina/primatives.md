
# **Kasmina Primitives Design Document**

## **Tier 1: Foundational Primitives (The "Obvious" Essentials)**

These are the non-negotiable building blocks required to replicate most standard neural network architectures. The goal here is completeness and correctness.

* **`linear` / `matmul`:** The core of most layers. Takes an input and weights, produces an output.
* **`add` / `multiply` / `divide`:** Basic element-wise arithmetic for residual connections, gating, etc.
* **Activation Functions:**
  * **`relu`:** The most basic non-linearity.
  * **`sigmoid` / `tanh`:** For gating and older recurrent-style logic.
  * **`gelu` / `swish` (or `silu`):** More modern and performant activation functions are essential.
* **Reduction Operations:**
  * **`reduce_sum` / `reduce_mean` / `reduce_max`:** Essential for pooling, normalization, and loss calculations. Must support reduction along a specific axis.
* **Normalization Layers:**
  * **`layer_norm`:** A critical component of almost all modern Transformer architectures.
  * **`rms_norm`:** A simpler, more efficient alternative to LayerNorm that Karn should have as an option.
* **Tensor Manipulation:**
  * **`reshape` / `view`:** To change the logical shape of a tensor.
  * **`permute` / `transpose`:** To reorder tensor dimensions.
  * **`concat`:** To join tensors along an axis.
  * **`split`:** To break a tensor into chunks.
* **`convolution_1d`:** The standard 1D convolution operator, essential for sequence processing.

---

## **Tier 2: Advanced & Optimized Primitives (The "Novel" Building Blocks)**

These primitives represent common, but more complex, patterns found in state-of-the-art models. Providing these as single, optimized operations prevents Karn from having to rediscover them from scratch and ensures high performance.

* **`softmax`:** A fused, numerically stable softmax operation. This is far better than having Karn combine `exp`, `sum`, and `divide`, which is prone to instability.
* **`rotary_pos_embedding` (RoPE):** A primitive that takes an input tensor and a position index and applies rotary embeddings. This is the standard for modern LLMs and would be a powerful tool for Karn.
* **`fused_add_norm`:** A single primitive that performs `LayerNorm(x + residual)`. This is a common fusion that saves a memory read/write cycle and improves performance.
* **`low_rank_matmul`:** A primitive that implements a low-rank update: `output = X + (X @ Wa @ Wb)`. This would allow Karn to invent LoRA-like adaptations from first principles.
* **`top_k`:** A differentiable `top_k` operator. It would take a tensor of logits and return a tensor where all but the top K values are masked to negative infinity. This could be used to learn sparse, high-impact features.
* **`group_norm`:** An alternative normalization scheme that operates on groups of channels, useful for CNN-style architectures.

---

## **Tier 3: Experimental & Exotic Primitives (The "Out-of-the-Box" Ideas)**

These are speculative but powerful primitives designed to allow `Karn` to break out of standard architectural patterns and discover truly novel computational structures.

* **`telemetry_probe` (The Self-Monitoring Primitive):**
  * **What it does:** An identity function (`output = input`) whose only purpose is to have a **side effect**: it writes detailed statistics about the intermediate tensor passing through it to a special telemetry buffer.
  * **Why it's powerful:** `Karn` can learn to insert these probes into its own creations. This allows it to design blueprints that are inherently more observable and debuggable, potentially learning to place probes at points of high variance or instability.

* **`stateful_update` (The Recurrent Primitive):**
  * **What it does:** An operation that explicitly models recurrence. It takes two inputs, `input` and `state_in`, and produces two outputs, `output` and `state_out`. For example: `state_out = tanh(W @ input + U @ state_in)`.
  * **Why it's powerful:** This gives `Karn` the fundamental building block of an RNN. It could learn to evolve recurrent state-passing mechanisms inside what is otherwise a standard feed-forward layer, creating hybrid architectures automatically.

* **`dynamic_conv` (The Data-Dependent Primitive):**
  * **What it does:** A 1D convolution where the filter weights are not static but are **dynamically generated** based on the input. The primitive would take two tensors: the `input_sequence` and a `generated_weights` tensor.
  * **Why it's powerful:** `Karn` could learn to pair this with a small MLP that generates the convolution filter on-the-fly for each token, allowing the network to adapt its processing based on the local context of the data.

* **`gate_by_metric` (The Meta-Learning Primitive):**
  * **What it does:** A dynamic gate that routes data based on a live telemetry metric. Its signature would be `gate(input_A, input_B, metric_value, threshold)`. It would return `input_A` if `metric_value > threshold`, and `input_B` otherwise.
  * **Why it's powerful:** `Karn` could design a blueprint that dynamically changes its own computational path based on its runtime state. For example: "If the gradient variance from the previous training step was too high, use the simpler, more stable `input_B` path; otherwise, use the complex, high-performance `input_A` path."

* **`fourier_transform` / `ifft` (The Frequency-Domain Primitive):**
  * **What it does:** Primitives that can transform a sequence into the frequency domain (FFT) and back (iFFT).
  * **Why it's powerful:** This allows `Karn` to learn to solve problems by operating on frequencies rather than sequences. It could evolve architectures like FNet where spatial convolutions are replaced by Fourier transforms, and it could discover this pattern on its own.
