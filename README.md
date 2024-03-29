# Subsampled Riemannian trust-region (RTR) algorithms

Authors: [Hiroyuki Kasai]http://kasai.comm.waseda.ac.jp/kasai/) and [Bamdev Mishra](https://bamdevmishra.in/)

Last page update: June 20, 2019

Latest library version: 1.1.0 (see Release notes for more info)

<br />

Introduction
----------
The package contains a MATLAB code presented in the report "[Inexact trust-region algorithms on Riemannian manifolds](http://papers.nips.cc/paper/7679-inexact-trust-region-algorithms-on-riemannian-manifolds)" by Hiroyuki Kasai and Bamdev Mishra in NeurIPS2018 (NIPS2018).

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
|docs/                  - PDF files of NeurIPS paper and poster.
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
<br />
This script demonstrates a case study of using the proposed Subsampled-RTR algorithm (Sub-H-RTR) on a small PCA problem instance. 
Below is the results of an optimality gap vs. iteration number and time [sec]. 
<br /><br />

<img src="https://github.com/hiroyuki-kasai/Subsampled-RTR/blob/master/figs/demo.png" width="900">

<br />

Performance results (from paper):
----------

- **PCA problem**
<img src="https://github.com/hiroyuki-kasai/Subsampled-RTR/blob/master/figs/pca.png" width="900">

- **Matrix completion problem**
<img src="https://github.com/hiroyuki-kasai/Subsampled-RTR/blob/master/figs/mc.png" width="900">


Cite
----------
Please cite our paper if you use this code in your own work.

```
@conference{Kasai_NeurIPS_2018,
    Author = {Kasai, H. and Mishra, B.},
    Booktitle = {NeurIPS},
    Title = {Inexact trust-region algorithms on {R}iemannian manifolds},
    Year = {2018}
}
```

<br />

License
----------
(c) 2017-2019 Hiroyuki Kasai (kasai **at** is **dot** uec **dot** ac **dot** jp) and Bamdev Mishra (bamdevm **at** gmail **dot** com).


<br />

Confirmed operating environment: 
--------------------------------
- This code has been checked on Mac OS (64 bits) under
    - MATLAB 8.2.0.701 (R2013b)
    - MATLAB 9.1.0.441655 (R2016b)
    - MATLAB 9.4.0.813654 (R2018a)
- Note that RTRMC may not work properly in some environments due to its Mex-files. 

<br />

Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) (email: hiroyuki **dot** kasai **at** waseda **dot** jp).

<br />

Release Notes
--------------
* Version 1.1.0 (June 20, 2019)
    - Added initial codes.
* Version 1.0.0 (Nov. 24, 2018)
    - Initial version.
