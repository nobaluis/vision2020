## HW02 - Pedestrians detection

## Requirements

`cmake >= 2.8`

`opencv 4.X`

`fftw 3.X`


Also this project uses a copy of the LBP C++ library from Nourani-Vatani, you can get the latest version [here](https://github.com/nourani/LBP).

## Assignment descriptions

Train a pedestrian classifier with SVM. Using HOG, LBP and HOG+LBP descriptors. 

![](report/images/pedestrians_bb.png)

## Results

| Descriptor | Tamaño | % Positivos | % Negativos | % Total |
| ---------- | ------:|:-----------:|:-----------:|:-------:|
| $\text{HOG}_{4 \times 4}$ | 16740 | 98.82 | 98.75 | 98.36 |
| $\text{HOG}_{8 \times 8}$ | 3780 | 97.64 | 98.75 | 97.55 |
| $\text{LBP}_{8}$ | 256 | 97.05 | 96.87 | 95.91 |
| $\text{LBP-HF}_{8}$ | 38 | 95.88 | 99.37 | 96.73 |
| $\text{LBP-HF}_{16}$ | 138 | 97.05 | 99.37 | 97.55 |
| $\text{HOG}_{4 \times 4} + \text{LBP}_{8}$ | 16996 | 100 | 99.37 | 99.59 |
| $\text{HOG}_{4 \times 4} + \text{LBP-HF}_{8}$ | 16778 | 99.41 | 100 | 99.59 |
| $\text{HOG}_{4 \times 4} + \text{LBP-HF}_{16}$ | 16878 | 99.41 | 99.37 | 99.18 |
| $\text{HOG}_{8 \times 8} + \text{LBP}_{8}$ | 634 | 99.41 | 99.37 | 99.18 |
| $\text{HOG}_{8 \times 8} + \text{LBP-HF}_{8}$ | 416 | 99.41 | 100 | 99.59 |
| $\text{HOG}_{8 \times 8} + \text{LBP-HF}_{16}$ | 516 | 99.41 | 100 | 99.59 |

