# Model choice: 5-line justification

Fill the `<...>` values from `results/COMPARISON.md` after running your real data.

1. **Accuracy:** DistilBERT scored <acc> / <macro F1> vs the baseline's <acc> / <macro F1>, a gap of <x> points.
2. **Where it matters:** DistilBERT wins on <classes, e.g. short or ambiguous tickets>; the baseline is good enough on <classes with clear keywords, e.g. Billing, Shipping>.
3. **Speed:** baseline <x> ms/ticket vs DistilBERT <y> ms/ticket on <CPU/GPU>, about <z>x slower.
4. **Cost:** the baseline trains in <x> s on a laptop CPU and serves from a tiny container; DistilBERT needed <y> s of training, ~250 MB of weights, and more RAM/CPU (or a GPU) to hit the same throughput.
5. **Decision:** deploy <model> because <accuracy gain justifies the cost / the gain is too small to justify 10-50x latency and cost>; revisit if <volume or accuracy requirement changes>.
