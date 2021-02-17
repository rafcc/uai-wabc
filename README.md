# Approximate Bayesian Computation for Bezier Simplex Fitting
This repository provides source files for reproducing the above paper submitted to UAI2021 and experiments therein.


## Requirements
- Docker 1.13.0 or above
- Git 2.0.0 or above
- GNU Make 4.2.0 or above


## How to reproduce our results
`Dockerfile` is provided for the required software.

First, build an image on your machine.

```
$ git clone https://github.com/rafcc/uai-wabc.git
$ cd uai-wabc
$ docker build --build-arg http_proxy=$http_proxy -t .
```

Then, run a container:

```
$ docker run --rm -v $(pwd):/data -it rafcc/uai-wabc
```

In the container, the experimental results can be reproduced by the following commands:

```
$ cd src
$ python experiments.py
```

Then, you will get the following directories which include experimental results:

```
../results/
```
