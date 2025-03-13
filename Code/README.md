
# State-of-the-Art Tuners

Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with PromiseTune.

## General Tuners
 - Random: a commonly used random search strategy, which is simple to implement and performs well in some cases.
 - [SMAC]("https://github.com/automl/SMAC3"): an algorithm configuration method based on sequence model optimization, which deals with categorical parameters by constructing a random forest model to select promising configurations in the algorithm configuration space.
 - [GA]("https://github.com/jMetal/jMetalPy"): a genetic algorithm for optimal configurations using natural selection and cross-variance heuristics.
 - [MBO]("https://github.com/PKU-DAIR/open-box"): a bayesian model which constructs the mixed kernel gausian process model to predict the objective function and uses the model to guide the search.


## Configuration Tuners
 - [FLASH]("https://github.com/FlashRepo/Flash-SingleConfig"): a sequential model-based approach that efficiently solves the single-objective configuration optimization problem for software systems and requires fewer measurements in the search for better configurations by using a priori knowledge of the configuration space to select the next promising configuration .
 - [Unicorn]("https://github.com/softsys4ai/unicorn"): an approach to analyze the performance of configurable systems through causal reasoning, which recovers the causal structure from performance data to help identify the root causes of performance failures, estimate parameter causal effects, and give recommendations for optimal configurations .

## Compiler Tuners
 - [BOCA]("https://github.com/BOCA313/BOCA"): the first automatic compiler tuning method based on Bayesian optimization, which designs novel search strategies by approximating the objective function using a tree-based model.
 - [CFSCA]("https://github.com/zhumxxx/CFSCA"): a compiler auto-tuning technique based on key flags selection, which determines potentially relevant flags by analyzing the program structure and compiler documentation, and then identifies the key flags by statistical analysis to narrow down the search space, so as to select an optimized sequence for the target program to improve performance.

## Database Tuners
 - [LlamaTune]("https://github.com/uw-mad-dash/llamatune"): a tool for configuration tuning of database management systems that utilizes techniques such as stochastic low-dimensional projection, special value bias sampling, and knob-value bucketing to reduce the search space.

 - [OtterTune]("https://github.com/cmu-db/ottertune"): an automated database management system tuning tool that combines supervised and unsupervised machine learning methods to optimize database configurations by reusing previous tuning data to select important configuration knobs, map workloads, and recommend settings.