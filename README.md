# Hopular: Modern Hopfield Networks for Tabular Data

_Bernhard Schäfl<sup>1</sup>, Lukas Gruber<sup>1</sup>,
Angela Bitto-Nemling<sup>1, 2</sup>, Sepp Hochreiter<sup>1, 2</sup>_

<sup>1</sup> ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria  
<sup>2</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)

---

##### Detailed blog post on this paper at [this link](https://ml-jku.github.io/hopular/).

---

While Deep Learning excels in structured data as encountered in vision and natural language processing, it failed to
meet its expectations on tabular data. For tabular data, Support Vector Machines (SVMs), Random Forests, and Gradient
Boosting are the best performing techniques with Gradient Boosting in the lead. Recently, we saw a surge of Deep
Learning methods that were tailored to tabular data but still underperformed compared to Gradient Boosting on
small-sized datasets. We suggest "Hopular", a novel Deep Learning architecture for medium- and small-sized datasets,
where each layer is equipped with continuous modern Hopfield networks. The modern Hopfield networks use stored data to
identify feature-feature, feature-target, and sample-sample dependencies. Hopular's novelty is that every layer can
directly access the original input as well as the whole training set via stored data in the Hopfield networks.
Therefore, Hopular can step-wise update its current model and the resulting prediction at every layer like standard
iterative learning algorithms. In experiments on small-sized tabular datasets with less than 1,000 samples, Hopular
surpasses Gradient Boosting, Random Forests, SVMs, and in particular several Deep Learning methods. In experiments on
medium-sized tabular data with about 10,000 samples, Hopular outperforms XGBoost, CatBoost, LightGBM and a state-of-the
art Deep Learning method designed for tabular data. Thus, Hopular is a strong alternative to these methods on tabular
data.

The full paper is available at: [https://arxiv.org/abs/2206.00664](https://arxiv.org/abs/2206.00664).

## Requirements

The software was developed and tested on the following 64-bit operating systems:

- Rocky Linux 8.5 (Green Obsidian)
- macOS 12.4 (Monterey)

As the development environment, [Python](https://www.python.org) 3.8.3 in combination
with [PyTorch Lightning](https://www.pytorchlightning.ai) 1.4.9 was used. More details on
how to install PyTorch Lightning are available on the [official project page](https://www.pytorchlightning.ai).

## Installation

The recommended way to install the software is to use `pip/pip3`:

```bash
$ pip3 install git+https://github.com/ml-jku/hopular
```

## Usage

Hopular has two modes of operation:

- `list` for displaying various information.
- `optim` for optimizing Hopular using specified hyperparameters.

More information regarding the operation modes is accessible via the `-h` flag (or, alternatively, by `--help`).

```bash
$ hopular -h
```

```bash
$ hopular <mode> -h
```

To display all available datasets, the `--datasets` flag has to be specified in the `list` mode.

```bash
$ hopular list --datasets 
```

Optimizing a Hopular model using the default hyperparameters is achieved by specifying the corresponding dataset in the
`optim` mode.

```bash
$ hopular optim --dataset <dataset_name>
```

## Examples

To optimize a Hopular model on the `GlassIdentificationDataset` using the default hyperparameters, only the dataset
name itself needs to be specified. More details on the default values are available in the
[console interface](hopular/interactive.py) implementation.

```bash
$ hopular optim --dataset "GlassIdentificationDataset"
```

Optimizing a smaller Hopular model on the `GlassIdentificationDataset` utilizing only `4` modern Hopfield networks, `2`
iterative refinement blocks, and a scaling factor of `10` is achieved by manually specifying the respective
hyperparameters.

```bash
$ hopular optim --dataset "GlassIdentificationDataset" --num_heads 4 --num_blocks 2 --scaling_factor 10
```

## Disclaimer

The [datasets](hopular/auxiliary/resources), which are part of this repository, are publicly available and may be
licensed differently. Hence, the [LICENSE](LICENSE) of this repository does not apply to them. More details on the
origin of the datasets are available in the accompanying paper.

## License

This repository is MIT-style licensed (see [LICENSE](LICENSE)), except where noted otherwise.