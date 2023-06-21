# collection of useful benchmarks
from __future__ import annotations

from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import time
# used for nvitop callback
from lightning.pytorch.utilities import rank_zero_only  # pylint: disable=import-error
from lightning.pytorch.utilities.exceptions import (  # pylint: disable=import-error
    MisconfigurationException,
)
from nvitop.api import libnvml
from nvitop.callbacks.utils import get_devices_by_logical_ids, get_gpu_stats
from nvitop import Device


class SamplesPerSecondBenchmark(Callback):
    def __init__(self, batch_interval=50):
        super().__init__()
        self.batch_interval = batch_interval    # defines the limit of batches for when the metric is computed the next time
        self.start_time = None
        self.num_samples = 0
        #self.batches_seen = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.num_samples = 0
        self.batch_interval = min(self.batch_interval, len(trainer.train_dataloader)//10)   # set the observation interval to 10% of the batches of one epoch
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # calculate how many samples have been passed so far
        batch_size = batch["input_ids"].size(dim=1) 
        self.num_samples += batch_size

        # compute and log the samplesPerSecond based on the number of samples
        if (batch_idx+1) % self.batch_interval == 0:  # log every batch_interval batches
            curr_time = time.time()
            elapsed_time = curr_time - self.start_time
            samplesPerSecond = self.num_samples / elapsed_time
            trainer.logger.experiment.log({"SamplesPerSecond": samplesPerSecond})


class GpuMetricsBenchmark(Callback):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ) -> None:
        super().__init__()

        try:
            libnvml.nvmlInit()
        except libnvml.NVMLError as ex:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback because NVIDIA driver is not installed.',
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature

    def on_train_start(self, trainer, pl_module) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback with Trainer that has no logger.',
            )

        if trainer.strategy.root_device.type != 'cuda':
            raise MisconfigurationException(
                f'You are using GpuStatsLogger but are not running on GPU. '
                f'The root device type is {trainer.strategy.root_device.type}.',
            )

        #device_ids = trainer.data_parallel_device_ids
        try:
            #self._devices = get_devices_by_logical_ids(device_ids, unique=True)
            self._devices = Device.cuda.all()
        except (libnvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                f'Cannot use GpuStatsLogger callback because devices unavailable. '
                f'Received: `gpus={device_ids}`',
            ) from ex

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_inter_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_intra_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    def _get_gpu_stats(self) -> dict[str, float]:
        """Get the gpu status from NVML queries."""
        return get_gpu_stats(
            devices=self._devices,
            memory_utilization=self._memory_utilization,
            gpu_utilization=self._gpu_utilization,
            fan_speed=self._fan_speed,
            temperature=self._temperature,
        )

