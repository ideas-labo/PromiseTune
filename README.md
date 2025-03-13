
# PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning

This repository contains the data and code for the following paper: 
> PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning

## Introduction
> The high configurability of modern software systems has made configuration tuning a crucial step for assuring system performance, e.g., latency or throughput. However, given the expensive measurements, large configuration space, and rugged configuration landscape, existing tuners suffer ineffectiveness due to the difficult balance of budget utilization between exploring uncertain regions (for escaping from local optima) and exploiting guidance of known good configurations (for fast convergence). The root cause is that we lack knowledge of where the **promising regions** lay, which also causes challenges in the explainability of the results.

> In this paper, we propose **PromiseTune** that tunes configuration guided by casually purified rules. PromiseTune is unique in the sense that we learn rules, which reflect certain regions in the configuration landscape, and purify them with causal inference. The remaining rules serve as approximated reflection of the promising regions, bounding the tuning to emphasize on these places in the landscape. This, as we demonstrate, can effectively mitigate the impact of exploration and exploitation trade-off. Those purified regions can then be paired with the measured configurations to provide spatial explainability at the landscape level. We evaluate PromiseTune against 10 state-of-the-art tuners on 12 systems and varying budgets, revealing that it performs significantly better than the others on 86% cases while providing richer information to explain the hidden system characteristics.


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
   - Data => Datasets, the target need to start with "$<"<br>
   - util => Util for tuners<br>
   - requirements.txt => Essential requirments need to be installed <br>
   - PromiseTune.py => The reproduction code of PromiseTune

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
If you have your own dataset, put the CSV file in "Code/Data/" and the target need to start with "$<". The default is minimize the target, if maximize please reverse the target. Finally, add it to the datasets list and run PromiseTune.


## <a name='tuners'></a>State-of-the-Art Tuners

Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with PromiseTune.

### General Tuners
 - Random: a commonly used random search strategy, which is simple to implement and performs well in some cases.
 - [SMAC](https://github.com/automl/SMAC3): a sequential model-based optimization, which deals with categorical parameters by constructing a random forest model to select promising configurations in the algorithm configuration space.
 - [GA](https://github.com/jMetal/jMetalPy): a genetic algorithm for optimal configurations using natural selection and cross-variance heuristics.
 - [MBO](https://github.com/PKU-DAIR/open-box): a bayesian model-based approach which constructs the mixed kernel gausian process model to predict the objective function and uses the model to guide the search.


### Configuration Tuners
 - [FLASH](https://github.com/FlashRepo/Flash-SingleConfig): a sequential model-based approach that efficiently solves the single-objective configuration optimization problem for software systems and requires fewer measurements in the search for better configurations by using a priori knowledge of the configuration space to select the next promising configuration.
 - [Unicorn](https://github.com/softsys4ai/unicorn): an approach to analyze the performance of configurable systems through causal reasoning, which recovers the causal structure from performance data to help identify the root causes of performance failures, estimate parameter causal effects, and give recommendations for optimal configurations.

### Compiler Tuners
 - [BOCA](https://github.com/BOCA313/BOCA): the first automatic compiler tuning method based on Bayesian optimization, which designs novel search strategies by approximating the objective function using a tree-based model.
 - [CFSCA](https://github.com/zhumxxx/CFSCA): a compiler auto-tuning technique based on key flags selection, which determines potentially relevant flags by analyzing the program structure and compiler documentation, and then identifies the key flags by statistical analysis to narrow down the search space, so as to select an optimized sequence for the target program to improve performance.

### Database Tuners
 - [LlamaTune](https://github.com/uw-mad-dash/llamatune): a tool for configuration tuning of database management systems that utilizes techniques such as stochastic low-dimensional projection, special value bias sampling, and knob-value bucketing to reduce the search space.

 - [OtterTune](https://github.com/cmu-db/ottertune): an automated database management system tuning tool that combines supervised and unsupervised machine learning methods to optimize database configurations by reusing previous tuning data to select important configuration knobs, map workloads, and recommend settings.


## RQ reproduction
- **RQ1 Effectiveness**: To measure the effectiveness of our method, you can directly run Quick Start. The other SOTA methods being compared are described in [State-of-the-Art Tuners](#tuners).

- **RQ2 Ablation**: Compare the differences when the key component rules are turned on or off:

   ```Set line 262 with 'rule = False'.```

- **RQ3 Sensitivity**: Analyze the sensitivity of the key parameter l, and set it to 5 or 10 or 15 or 20:

   ```Set line 263 with 'l = 5'.```

- **RQ4 Explainability Case Study**: Conduct a case study on the explainability of the method using the system **x264** as an example. 
   ```Set line 272 with 'systems = ['x264']' ```

##  RQ supplementary
RQ_supplementary contains the specific supplementary files for Table2 and Table3 of our paper.