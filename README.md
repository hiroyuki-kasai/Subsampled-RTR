# Subsampled Riemannian trust-region (RTR) algorithms

Authors: [Hiroyuki Kasai](http://www.kasailab.com/kasai) and [Bamdev Mishra](https://bamdevmishra.in/)

Last page update: June 20, 2019

Latest library version: 1.1.0 (see Release notes for more info)

<br />

Introduction
----------
The package contains a MATLAB code presented in the report "Inexact trust-region algorithms on Riemannian manifolds" by Hiroyuki Kasai and Bamdev Mishra in NeurIPS2018 (NIPS2018).

<br />

Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script to check and understand this package easily. 
|manopt_mod_solvers/    - Contains modified solvers from manopt.
|manopt-4/              - Contains manopt package.
|problem/               - Problem definition files to be solved.
|propose/               - Contains proposed algorithm files.
|tools/                 - Contains data generation scripts.
|docs/                  - PDF files of NeurIPS paper.
</pre>   

<br />

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

<br />

Demonstration
----------------------------
Run `demo` for a simple demonstartion. 
```Matlab
%% Run a demonstration script
demo; 
```

This script demonstrates the case study of using the proposed Subsampled-RTR algorithm (Sub-H-RTR) on a small PCA problem instance. 
Below is the results of the optimality gap vs. iteration number and time [sec]. 


<img src="https://github.com/hiroyuki-kasai/Subsampled-RTR/figs/demo.png" width="900">
<br /><br />

<br />

Document
----------
The document can be obtained from below;

- H. Kasai and B. Mishra, "[Inexact trust-region algorithms on Riemannian manifolds](https://neurips.cc/Conferences/2018/Schedule?showEvent=11421)," NeurIPS2018 (NIPS2018).

<br />

License
----------
(c) 2017-2018 Hiroyuki Kasai (kasai **at** is **dot** uec **dot** ac **dot** jp) and Bamdev Mishra (bamdevm **at** gmail **dot** com).


<br />

Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://www.kasailab.com/kasai_e.htm) (email: kasai **at** is **dot** uec **dot** ac **dot** jp).

<br />

Release Notes
--------------
* Version 1.1.0 (June 20, 2019)
    - Added initial codes.
* Version 1.0.0 (Nov. 24, 2018)
    - Initial version.
