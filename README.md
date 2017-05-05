# skopt-hpolib2
Scripts to run skopt optimizers on HPOlib's benchmark problems

# Instructions
1. Install the huge set of requirements
```
pip install -r requirements.txt
```
2. Install scikit-optimize master
```
pip install git+http://github.com/scikit-optimize/scikit-optimize.git
```

3. Install HPOlib2 development branch
```
pip install git+https://github.com/automl/HPOlib2.git@development
```

4. Run benchmark problems using

```
python3 bench_ml.py --optimizer="gp" --problem="lr"
```
