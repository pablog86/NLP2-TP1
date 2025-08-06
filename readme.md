# Trabajo Práctico: TinyGPT y Mixture of Experts (MoE)

## 📚 Descripción

Este repositorio contiene la implementación de **TinyGPT**, un modelo Transformer de práctica, y su extensión a **Mixture of Experts (MoE)**. El objetivo es:

- **Comparar** un GPT denso vs. un MoE en cuanto a:

  - Arquitectura y número de parámetros.
  - Rendimiento en entrenamiento (soft-mixture + aux-loss) e inferencia (hard top-1 routing).
  - Métricas de calidad: Perplejidad en set de validación.
  - Throughput (tokens/s) y uso de memoria.

- **Experimentar** con técnicas de generación de texto: greedy, temperatura, top-k y top-p (nucleus sampling).

## 🗂 Estructura de Carpetas

```
├── TinyGPT.ipynb           # Notebook principal con todos los pasos
├── trainer.py              # Lógica de entrenamiento y evaluación
├── checkpoints/            # Checkpoints guardados
│   └── dense/              # Modelo TinyGPT con FFN
│   └── moe/                # Modelo TinyGPT MoE
├── runs/                   # Checkpoints guardados
│   └── experiment1/        # Tensorboard para TinyGPT con FFN
│   └── experiment2/        # Tensorboard para TinyGPT MoE
└── README.md               # Este archivo
```

## 🚀 Uso

### 1. Entrenamiento

```python
writer = SummaryWriter(log_dir="./runs/<...>")

trainer = Trainer(
    model=<...>,
    train_data_loader=train_loader,
    test_data_loader=val_loader,
    loss_fn=loss_fn,
    gradient_accumulation_steps=grad_accum,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    save_dir="./checkpoints/<...>",
    save_every_n=500
)
```

- Guarda checkpoints en `checkpoints/`.
- Registra métricas con TensorBoard en `runs/`.

### 2. Inferencia y Generación

```python
from utils import generateV2, encode, decode
import torch

# Cargar modelo
model = model(config).to(device)
model.load_state_dict(torch.load('models/model_final.pt'))
model.eval()

# Generar texto
output, throughput = generateV2(
    model=model,
    device=device,
    prompt="To be or not to be,",
    max_new_tokens=100,
    temperature=0.7,
    top_k=10,
    top_p=0.9
)
print(output)
print(f"Throughput: {throughput:.1f} tokens/s")
```

### 3. Evaluación de Perplejidad

- Throughput desde la inferencia con generateV2 (GPU y CPU)
- Cantidad de parámetros
- Perplejidad
    ```python
    from trainer import compute_perplexity
    import torch.nn as nn

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    dense_ppl = compute_perplexity(dense_model, val_loader, loss_fn, device)
    moe_ppl   = compute_perplexity(moe_model,   val_loader, loss_fn, device)
    print(f"Dense PPL: {dense_ppl:.2f}")
    print(f"MoE   PPL: {moe_ppl:.2f}")
    ```

## 📈 Resultados y Benchmark

- **Parámetros**: Dense vs MoE (4 expertos × capa FFN)
- **Perplejidad** en validación: `ppl_dense` vs `ppl_moe`
- **Throughput** (tokens/s) en GPU y CPU
- **Curvas** de Train Loss / Val Loss / Learning Rate (ver `notebooks/Graphs.ipynb`)

## 📑 Conclusiones

- MoE aumenta la capacidad sin incrementar linealmente la latencia en inferencia *si* el modelo es lo suficientemente grande o el routing está optimizado.
- En toy-models pequeños, el overhead de routing puede superar la ganancia de ejecutar un solo experto.
- Métricas de perplejidad y throughput muestran trade-offs claros entre calidad y velocidad.

## 📖 Referencias

## 📖 Referencias

- **Switch Transformer** (Fedus et al., 2021) — *Efficient Mixture of Experts* ([https://arxiv.org/abs/2201.00084](https://arxiv.org/abs/2201.00084))
- **ST-MoE** (Zoph et al., 2022) — *Scaling MoE Models* ([https://arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906))
- **A Review of Sparse Expert Models in Deep Learning** (Wright et al., 2022) — *Sparse Expert Models* ([https://arxiv.org/pdf/2209.01667](https://arxiv.org/pdf/2209.01667))
- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2019) — *The Curious Case of Neural Text Degeneration* ([https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751))
- **Closing the Curious Case of Neural Text Degeneration** (Finlayson et al., 2023) — *Closing the Curious Case of Neural Text Degeneration* ([https://openreview.net/pdf?id=dONpC9GL1o](https://openreview.net/pdf?id=dONpC9GL1o))
- **Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation** (Xu et al., 2022) — *Learning to Break the Loop* ([https://arxiv.org/pdf/2206.02369](https://arxiv.org/pdf/2206.02369))
- **On Decoding Strategies for Neural Text Generators** (Wiher et al., 2022) — *On Decoding Strategies for Neural Text Generators* ([https://arxiv.org/pdf/2203.15721](https://arxiv.org/pdf/2203.15721))


## 📝 Licencia

Este código esta basado en la especialización de CEIA de FIUBA, repo de la materia [NLP2](https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/CEIA-LLMIAG/blob/main/ClaseIV/TinyGPT.ipynb) *Author: Abraham R.*

