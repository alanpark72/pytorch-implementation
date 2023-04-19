import os
import yaml

def getConfig(path="./config/sample_train.yaml"):
    with open(path, 'r') as f:
        config=yaml.load(f, yaml.FullLoader)
    
    return config

def checkDir(path, auto_increment=False):
    if os.path.exists(path):
        if auto_increment:
            for cnt in range(1,9999):
                _path = path[:-1]+"_{:02d}/".format(cnt)
                if not os.path.exists(_path):
                    os.makedirs(_path)
                    return _path
        else:
            return path
    else:
        os.makedirs(path)
        return path