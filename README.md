# The T-GOSPA metric

This repository contains the Python implementation of the  trajectory generalised optimal subpattern assignment (T-GOSPA) metric for evaluation of multi-object tracking algorithms proposed in \[1].

The T-GOSPA metric is a mathematically principled metric for sets of trajectories that penalises localisation errors for properly detected objects, the number of false objects, the number of missed objects and the number of track switches. The implementation is based on linear programming. This repository also contains the Python implementation of the time-weighted T-GOSPA metric \[2].

These two T-GOSPA metrics are an extension of the GOSPA metric for sets of objects \[3]\[4].



The repository also includes the Python implementation of the T-GOSPA quasi-metric (q-metric), which admits unequal costs for missed and false objects \[5].





To see examples of the T-GOSPA metric, run the file

examples\_TGOSPA\_metric.py

To see examples of the time-weighted T-GOSPA metric, run the file

examples\_TW\_TGOSPA\_metric.py

To see examples of the T-GOSPA q-metric, run the file

examples\_TGOSPA\_qmetric.py





\[1] A. F. Garcia-Fernandez, A. S. Rahmathullah and L. Svensson, "A Metric on the Space of Finite Sets of Trajectories for Evaluation of Multi-Target Tracking Algorithms," in IEEE Transactions on Signal Processing, vol. 68, pp. 3917-3928, 2020, doi: 10.1109/TSP.2020.3005309
https://arxiv.org/abs/1605.01177

\[2] A. F. Garcia-Fernandez, A. S. Rahmathullah and L. Svensson, "A time-weighted metric for sets of trajectories to assess multi-object tracking algorithms," 2021 IEEE 24th International Conference on Information Fusion (FUSION), Sun City, South Africa, 2021, pp. 1-8, doi: 10.23919/FUSION49465.2021.9626977
https://arxiv.org/abs/2110.13444

\[3] A. S. Rahmathullah, A. F. Garcia-Fernandez and L. Svensson, "Generalized optimal sub-pattern assignment metric," 2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8, doi: 10.23919/ICIF.2017.8009645.
https://arxiv.org/abs/1601.05585

\[4] https://github.com/ewilthil/gospapy

\[5] A. F. Garcia-Fernandez, J. Gu, L. Svensson, Y. Xia, J. Krejc, O. Kost, O. Straka "GOSPA and T-GOSPA quasi-metrics for evaluation of multi-object tracking algorithms," in IEEE Transactions on Aerospace and Electronic Systems, 2026.

