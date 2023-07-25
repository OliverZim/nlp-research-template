import wandb
import numpy as np

wandb.login()
run = wandb.Api().run(path="oliver-zimmermann/benchmarks/3ki68g7q")

power_used_result = run.scan_history()
values = []
for value in power_used_result:
    value = value['gpu_id: 0/power.relative (%)']
    if(value!=None):
        values.append(value)
print(values)
print("MEAN:")
print(np.mean(values))