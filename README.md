[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/YQHnUjoB)
# Sudoku Solver Distributed System


## How to start nodes
Run in two different terminal or more(you can start more then 2 nodes):

Start the main node:
```console
$ python3 NodeSolver.py -p 8000 -s 9000
```
Start a node and connect to another node:
```console
$ python3 NodeSolver.py -p 8001 -s 9001 -a localhost:9000
```
## How to run
Start the solving process:
```console
$ curl http://localhost:8000/solve -X POST -H 'Content-Type: application/json' -d '{"sudoku": [[3, 4, 5, 6, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0], [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6], [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]]}'
```

Watch the network status:
```console
$ curl http://localhost:8001/network
```

Watch the stats of all nodes:
```console
$ curl http://localhost:8001/stats
```


