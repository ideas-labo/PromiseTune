
# PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning

This repository contains the data and code for the following paper: 
> PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning

## Introduction
The high configurability of modern software systems has made configuration tuning a crucial step for assuring system performance, e.g., latency or throughput. However, given the expensive measurements, large configuration space, and rugged configuration landscape, existing tuners suffer ineffectiveness due to the difficult balance of budget utilization between exploring uncertain regions (for escaping from local optima) and exploiting guidance of known good configurations (for fast convergence). The root cause is that we lack knowledge of where the **promising regions** lay, which also causes challenges in the explainability of the results.

In this paper, we propose **PromiseTune** that tunes configuration guided by casually purified rules. PromiseTune is unique in the sense that we learn rules, which reflect certain regions in the configuration landscape, and purify them with causal inference. The remaining rules serve as approximated reflection of the promising regions, bounding the tuning to emphasize on these places in the landscape. This, as we demonstrate, can effectively mitigate the impact of exploration and exploitation trade-off. Those purified regions can then be paired with the measured configurations to provide spatial explainability at the landscape level. We evaluate PromiseTune against 10 state-of-the-art tuners on 12 systems and varying budgets, revealing that it performs significantly better than the others on 86% cases while providing richer information to explain the hidden system characteristics.


## Systems

| System     | Version | Benchmark    | Domain            | Language | Performance         | $B/N$ | S_space    |
| ---------- | ------- | -------------------------------------------------- | ----------------- | -------- | ------------------- | ------------------------------ | ------------------------- |
| 7z         | 9.20    | Compressing a 3 GB directory                       | File Compressor   | C++      | Runtime (ms)        | 11/3                           | $1.68 \times 10^8$         |
| DConvert   | 1.0.0   | Transform resources at different scales            | Image Scaling     | Java     | Runtime (s)         | 17/1                           | $1.05 \times 10^7$         |
| ExaStencils| 1.2     | Default benchmarks                                 | Code Generator    | Scala    | Runtime (ms)        | 7/5                            | $1.61 \times 10^9$         |
| BDB-C      | 18.0    | Benchmark provided by vendor                       | Database          | C        | Latency (s)         | 16/0                           | $6.55 \times 10^4$         |
| DeepArch   | 2.2.4   | UCR Archive time series dataset                    | Deep Learning Tool| Python   | Runtime (min)       | 12/0                           | $4.10 \times 10^3$         |
| PostgreSQL | 22.0    | PolePosition 0.6.0                                 | Database          | C        | Runtime (ms)        | 6/3                            | $1.42 \times 10^9$         |
| JavaGC     | 7.0     | DaCapo benchmark suite                             | Java Runtime      | Java     | Runtime (ms)        | 12/23                          | $2.67 \times 10^{41}$      |
| Storm      | 0.9.5   | Randomly generated benchmark                       | Data Analytics    | Clojure  | Messages per Second | 12/0                           | $4.10 \times 10^{3}$       |
| x264       | 0.157   | Video files of various sizes                       | Video Encoder     | C        | Peak signal-to-noise ratio | 4/13                   | $6.43 \times 10^{26}$      |
| Redis      | 6.0     | Sysbench                                           | Database          | C        | Requests per second | 1/8                            | $5.78 \times 10^{16}$      |
| HSQLDB     | 19.0    | PolePosition 0.6.0                                 | Database          | Java     | Runtime (ms)        | 18/0                           | $2.62 \times 10^5$         |
| LLVM       | 3.0     | LLVM’s test suite                                  | Compiler          | C++      | Runtime (ms)        | 10/0                           | $1.02 \times 10^3$         |

## Code Structure
   - Data (Datasets, the target need to start with "$<")<br>
   - util (Util for tuners)<br>
   - requirements.txt (Essential requirments need to be installed) <br>
   - PromiseTune (The reproduction code of PromiseTune)

## Quick Start

* Python 3.8+

To run the code, and install the essential requirements: 
```
pip install -r requirements.txt
```

And you can run the below code to have a quick start:
```
cd ./Code
python3 PromiseTune.py
```
If you have your own dataset, put the CSV file in "Code/Data/" and the target need to start with "$<". Finally, add it to the datasets list and run PromiseTune.

##  RQ Supplementary
RQ_supplementary contains the supplementary files for our research finds.