# HighEffiNNI

This repository is a high-efficiency training framework with [NNI](https://github.com/microsoft/nni) for PyTorch. 

## Requirements

To install requirements:

```setup
PyTorch >= 1.6
nni >= 2.8
easydict >= 1.9
```
## config.yml
A example of `config.yml` can be found in this repo.
More experiment config references can see [here](https://nni.readthedocs.io/en/stable/reference/experiment_config.html).

## Running

To start up the nni, run this command:

```
nnictl create --config config.yml -p 8140
```

To watch the running of nni, run this command:

```
nnictl top 
```


To stop the nni, run this command:

```
nnictl stop [--all] ([id])
```

TODO:
- [ ] To integrate the distributed training framework horovod into NNI.
- [ ] To develop more efficient training skills.
