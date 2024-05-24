# UNITE

* Create virtual environment through anaconda
```
conda create --name UNITEEnv python=3.8
conda activate UNITEEnv
```
* Install packages
   
```
pip install -r requirements.txt
```

* Install [MAESTRO](https://github.com/maestro-project/maestro.git)
```
python build.py
```


### Run UNITE at the first stage ###

```
sh run.sh
```

### Run UNITE at the second stage ###
```
sh run2.sh
```

### Run UNITE at the third stage ###
```
sh run3.sh
```