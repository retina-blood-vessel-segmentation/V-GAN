import glob
import mlflow
import os
import utils
from pathlib import Path

project_path = Path('.').resolve()

def predict_parallel():
    root = Path('..')
    threads = 8
    for d in ['CHASE']:
        allimgs = utils.all_files_under(root / "data" / "eval" / d / "images")
        allmasks = utils.all_files_under(root / "data" / "eval" / d / "masks", extension="png")
        jobs = []
        assert(len(allimgs) == len(allmasks))
        n = len(allimgs)
        s = n // threads
        for i in range(threads):
            lower = i * s
            upper = min(s*(i+1), n)
            img = allimgs[lower:upper]
            masks = allmasks[lower:upper]
            with open(f'{i}.tmp','w') as f:
                for j in range(len(img)):
                    f.write(f'{img[j]}\n{masks[j]}\n')
            p = {
                    'dataset' : d,
                    'spec' : f'{i}.tmp' 
            }
            print(f"Starting job {i}")
            jobs.append(
                mlflow.projects.run(
                    uri=str(project_path),
                    entry_point='test',
                    parameters=p,
                    experiment_name=f'VGAN-{i}',
                    use_conda=False,
                    synchronous=False
                )
            )
        for j in jobs:
            j.wait()

if __name__ == '__main__':
    predict_parallel()   