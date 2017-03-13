# Usage

## Environment
- Python, 3.5+
- scikit-learn, 0.18.1

## Running Examples

### 1. Predict the result of small dataset

Change the corrsponding parameters in _main.py_ to small.

```python
# To run the small dataset prediction, assign the corresponding file name as parameter
if __name__ == '__main__':
    predict(scale='small', output_file='output-small.txt')
```

In terminal, call the following command: 

```command
$ python3 main.py
```

### 2. Predict the result of large dataset

Change the corrsponding parameters in _main.py_ to large.

```python
# To run the small dataset prediction, assign the corresponding file name as parameter
if __name__ == '__main__':
    predict(scale='large', output_file='output-large.txt')
```

In terminal, call the following command: 

```command
$ python3 main.py
```

### 3. Skip the grid search process

Normally, we run the grid search to obtain the best params first in our prediction in #1 and #2 exmaple. However, it may take a long time, so you can just skip this process by assigning the specific _C_, _gamma_, _kernel combination_.

Recall that, he combination of _C_, _gamma_, kernel is {'gamma': 0.0078125, 'C': 16, 'kernel': 'rbf'} performs best.


```python
if __name__ == '__main__':
    predict(scale='small', output_file='output-small.txt', C=16, gamma=0.0781, kernel='rbf')
```