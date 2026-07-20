from copy import deepcopy

import pathos
import pytest
import deltakit_stim as stim
from deltakit_core.decoding_graphs import dem_to_decoding_graph_and_logicals

from deltakit_decode import PyMatchingDecoder
from deltakit_decode.analysis._matching_decoder_managers import GraphDecoderManager
from deltakit_decode.noise_sources import FixedWeightMatchingNoise, ToyNoise


class MockNoiseModelDecoderManagerPoolMap:
    """Class for mocking process pool map call to decoder manager thread workers to
    execute in main process by patching _decoder_manager.mp_dm."""

    def __init__(self, proc_managers, monkeypatch):
        self.proc_managers = proc_managers
        self.monkeypatch = monkeypatch

    def map(self, func, iterable):
        results = []
        for proc_id, args in enumerate(iterable):
            self.monkeypatch.setattr(
                "deltakit_decode.analysis._decoder_manager.mp_dm",
                self.proc_managers[proc_id],
            )
            results.append(func(args))

        return results


class TestNoiseModelDecoderManager:
    @pytest.fixture(scope="class")
    @classmethod
    def stim_circuit(cls):
        circuit = stim.Circuit.generated(
            "surface_code:unrotated_memory_z", rounds=3, distance=3
        )
        return ToyNoise(1e-3).permute_stim_circuit(circuit)

    @pytest.fixture(scope="class")
    @classmethod
    def surface_decoder(cls, stim_circuit):
        dem = stim_circuit.detector_error_model(decompose_errors=True)
        return PyMatchingDecoder(*dem_to_decoding_graph_and_logicals(dem))

    @pytest.fixture
    def noise_model(self):
        return FixedWeightMatchingNoise(13)

    @pytest.fixture(params=[GraphDecoderManager])
    def decoder_manager(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2**63])
    def seed(self, request) -> int:
        return request.param

    def test_run_batch_shots_parallel_worker_base_seeds_are_updated(
        self,
        decoder_manager,
        noise_model,
        seed,
        surface_decoder,
        monkeypatch,
    ):
        num_procs = 2
        batch_num = 12

        manager = decoder_manager(noise_model, surface_decoder, seed=seed)
        proc_1 = deepcopy(manager)
        proc_2 = deepcopy(manager)
        pool_map_mock = MockNoiseModelDecoderManagerPoolMap(
            proc_managers=[proc_1, proc_2], monkeypatch=monkeypatch
        )
        pool = pathos.multiprocessing.ProcessPool(nodes=num_procs)

        monkeypatch.setattr(pool, "map", pool_map_mock.map)

        seed_history = {manager.seed}
        if seed == 2**63:
            with pytest.raises(OverflowError):
                manager.run_batch_shots_parallel(
                    batch_limit=10,
                    processes=num_procs,
                    pool=pool,
                    min_tasks_per_process=1,
                )
        else:
            for _ in range(batch_num):
                manager.run_batch_shots_parallel(
                    batch_limit=10,
                    processes=num_procs,
                    pool=pool,
                    min_tasks_per_process=1,
                )
                assert proc_1.seed == proc_2.seed == manager.seed
                seed_history.add(manager.seed)
            # as seeds are chosen uniformly at random this has a probabilistic failure
            # case, but is extremely small (very roughly 1 in 10^16)
            assert len(seed_history) == batch_num + 1
            assert proc_1 is not proc_2

    def test_run_batch_shots_parallel_seeds_are_reproducible(
        self,
        decoder_manager,
        noise_model,
        seed,
        surface_decoder,
        monkeypatch,
    ):
        num_procs = 1
        batch_num = 12

        manager = decoder_manager(noise_model, surface_decoder, seed=seed)
        proc_1 = deepcopy(manager)
        pool_map_mock = MockNoiseModelDecoderManagerPoolMap(
            proc_managers=[
                proc_1,
            ],
            monkeypatch=monkeypatch,
        )
        pool = pathos.multiprocessing.ProcessPool(nodes=num_procs)
        monkeypatch.setattr(pool, "map", pool_map_mock.map)
        first_run_seeds = [manager.seed]
        for _ in range(batch_num):
            manager.run_batch_shots_parallel(
                batch_limit=10, processes=num_procs, pool=pool, min_tasks_per_process=1
            )
            assert manager.seed == proc_1.seed
            first_run_seeds.append(manager.seed)

        manager.reset()
        proc_1 = deepcopy(manager)
        pool_map_mock = MockNoiseModelDecoderManagerPoolMap(
            proc_managers=[
                proc_1,
            ],
            monkeypatch=monkeypatch,
        )
        monkeypatch.setattr(pool, "map", pool_map_mock.map)
        second_run_seeds = [manager.seed]
        for _ in range(batch_num):
            manager.run_batch_shots_parallel(
                batch_limit=10, processes=num_procs, pool=pool, min_tasks_per_process=1
            )
            assert manager.seed == proc_1.seed
            second_run_seeds.append(manager.seed)

        assert first_run_seeds == second_run_seeds
