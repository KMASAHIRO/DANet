import subprocess

subprocess.run(['pip','install','tensorflow==2.4.1'])
subprocess.run(['pip','install','museval==0.4.0'])
subprocess.run(['pip','install','soundfile==0.10.3.post1'])
subprocess.run(['pip','install','pandas==1.1.5'])
subprocess.run(['pip','install','numpy==1.19.5'])
subprocess.run(['pip','install','scipy==1.4.1'])
subprocess.run(['pip','install','librosa==0.8.0'])
subprocess.run(['pip','install','matplotlib==3.2.2'])

from . import preprocess
from . import generating_data
from . import models
from . import evaluation