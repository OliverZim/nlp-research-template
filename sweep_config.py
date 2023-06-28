SWEEP_CONFIG = dict(
    name="benchmarks",
    project="Benchmarking",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        wandb_run_name=dict(
            values=["benchmarks"],
        ),
        data_dir=dict(
            values=["./data/loremIpsum"],
        ),
        precision=dict(
            values=["16-mixed", "bf16-mixed", 32]
        ),
        workers=dict(
            values=[1,2,4,8,16,24,32]
        ),
        compile=dict(
            values=[True, False]
        ),
    ),
)


--wandb_run_name="benchmarks" -d ./data/loremIpsum