# HighPerfNNI

This repository is a high-performance training framework with NNI for PyTorch. 

## Requirements

To install requirements:

```setup
PyTorch >= 1.6
nni >= 2.8
```

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
