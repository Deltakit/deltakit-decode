# (c) Copyright Riverlane 2020-2026. All rights reserved.
from importlib.util import find_spec
from itertools import tee
from pathlib import Path

import deltakit_circuit as sp
import deltakit_stim as stim
import networkx as nx
import pytest
from deltakit_core.decoding_graphs import FixedWidthBitstring

from deltakit_decode.utils._graph_circuit_helpers import (
    get_irrelevant_nodes,
    get_trimmed_circuit,
    parse_stim_circuit,
    split_measurement_bitstring,
    stim_circuit_to_graph_dem,
)

try:
    from lestim import Circuit as StimCircuit
except ImportError:
    from stim import Circuit as StimCircuit


def test_stim_circuit_to_graph_dem_does_not_decompose_the_rep_code():
    stim_rep_code = stim.Circuit.generated(
        "repetition_code:memory",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.1,
    )

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find("^") == -1


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
def test_stim_circuit_to_graph_dem_does_decompose_non_rep_codes(code_task):
    stim_rep_code = stim.Circuit.generated(
        code_task, distance=5, rounds=5, after_clifford_depolarization=0.1
    )

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find("^") != -1


class TestSplitMeasurementBitstring:
    @pytest.mark.parametrize(
        ("stim_circuit", "measurement_bitstring", "expected_split_bitstring"),
        [
            (
                stim.Circuit.generated(
                    "surface_code:rotated_memory_x",
                    distance=3,
                    rounds=1,
                    after_clifford_depolarization=0.1,
                ),
                FixedWidthBitstring(17, 0b01011010001011011),
                [
                    FixedWidthBitstring(8, 0b01011011),
                    FixedWidthBitstring(9, 0b010110100),
                ],
            ),
            (
                stim.Circuit.generated(
                    "surface_code:rotated_memory_x",
                    distance=3,
                    rounds=3,
                    after_clifford_depolarization=0.1,
                ),
                FixedWidthBitstring(33, 0b110011101100101001010001110000000),
                [
                    FixedWidthBitstring(8, 0b10000000),
                    FixedWidthBitstring(8, 0b10100011),
                    FixedWidthBitstring(8, 0b10010100),
                    FixedWidthBitstring(9, 0b110011101),
                ],
            ),
        ],
    )
    def test_measurement_bitstring_can_be_split_by_each_layer_of_measurement_gates(
        self, stim_circuit, measurement_bitstring, expected_split_bitstring
    ):
        split_bitstring = split_measurement_bitstring(
            measurement_bitstring, stim_circuit
        )
        for split_bitstring_i, expected_split_bitstring_i in zip(
            split_bitstring, expected_split_bitstring
        ):
            assert split_bitstring_i == expected_split_bitstring_i

    @pytest.mark.parametrize(
        ("stim_circuit", "measurement_bitstring", "expected_number_of_bitstrings"),
        [
            (
                stim.Circuit.generated(
                    "surface_code:rotated_memory_x",
                    distance=3,
                    rounds=1,
                    after_clifford_depolarization=0.1,
                ),
                FixedWidthBitstring(17, 0b01011010001011011),
                2,
            ),
            (
                stim.Circuit.generated(
                    "surface_code:rotated_memory_x",
                    distance=3,
                    rounds=3,
                    after_clifford_depolarization=0.1,
                ),
                FixedWidthBitstring(33, 0b110011101100101001010001110000000),
                4,
            ),
        ],
    )
    def test_length_of_split_bitstring_matches_number_of_layers_with_measurement_gates(
        self, stim_circuit, measurement_bitstring, expected_number_of_bitstrings
    ):
        split_bitstring = split_measurement_bitstring(
            measurement_bitstring, stim_circuit
        )
        assert len(split_bitstring) == expected_number_of_bitstrings


def stim_circuit_rep_5x4():
    return stim.Circuit.generated(
        "repetition_code:memory",
        rounds=4,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def stim_circuit_rplanar_3x3x3():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def stim_circuit_planar_5x5x2():
    return stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=2,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


class TestParseStimCircuit:
    @pytest.fixture(
        params=[
            stim_circuit_rep_5x4(),
            stim_circuit_rplanar_3x3x3(),
            stim_circuit_planar_5x5x2(),
        ],
        scope="class",
    )
    @classmethod
    def stim_circuit(cls, request):
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def original_graph_trimmed_graph_logicals(cls, stim_circuit):
        trimmed_graph, logicals, _ = parse_stim_circuit(stim_circuit, trim_circuit=True)
        original_graph, _, _ = parse_stim_circuit(stim_circuit, trim_circuit=False)
        return original_graph, trimmed_graph, logicals

    def test_trimmed_stim_circuit_has_same_number_of_detectors_as_its_corresponding_trimmed_graph(
        self, stim_circuit
    ):
        trimmed_graph, _, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_detectors == len(trimmed_graph.nodes) - len(
            trimmed_graph.boundaries
        )

    def test_trimmed_stim_circuit_has_same_number_of_observables_as_its_corresponding_trimmed_graph(
        self, stim_circuit
    ):
        _, trimmed_logicals, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_observables == len(trimmed_logicals)

    def test_logicals_are_reachable_in_trimmed_stim_graph(
        self, original_graph_trimmed_graph_logicals
    ):
        _, trimmed_graph, logicals = original_graph_trimmed_graph_logicals
        non_boundary_nodes = (
            node
            for node in trimmed_graph.nodes
            if not trimmed_graph.detector_is_boundary(node)
        )
        for node in non_boundary_nodes:
            assert any(
                any(
                    nx.has_path(trimmed_graph.no_boundary_view, node, logical_a)
                    for logical_a, _ in logical
                    if not trimmed_graph.detector_is_boundary(logical_a)
                )
                or any(
                    nx.has_path(trimmed_graph.no_boundary_view, node, logical_b)
                    for _, logical_b in logical
                    if not trimmed_graph.detector_is_boundary(logical_b)
                )
                for logical in logicals
            )

    def test_logicals_edges_are_still_in_trimmed_graph(
        self, original_graph_trimmed_graph_logicals
    ):
        _, trimmed_graph, logicals = original_graph_trimmed_graph_logicals
        assert all(
            edge in trimmed_graph.edges for logical in logicals for edge in logical
        )

    def test_trimmed_graph_has_no_more_nodes_than_origin(
        self, original_graph_trimmed_graph_logicals
    ):
        original_graph, trimmed_graph, _ = original_graph_trimmed_graph_logicals
        assert len(trimmed_graph.nodes) <= len(original_graph.nodes)

    def test_trimmed_graph_has_no_more_edges_than_origin(
        self, original_graph_trimmed_graph_logicals
    ):
        original_graph, trimmed_graph, _ = original_graph_trimmed_graph_logicals
        assert len(trimmed_graph.edges) <= len(original_graph.edges)

    def test_detector_order_is_unchanged_without_lexical_detectors(self, stim_circuit):
        _, _, stim_circuit_out = parse_stim_circuit(
            stim_circuit, trim_circuit=False, lexical_detectors=False
        )
        circuit_in = sp.Circuit.from_stim_circuit(stim_circuit)
        circuit_out = sp.Circuit.from_stim_circuit(stim_circuit_out)
        assert len(circuit_in.detectors()) == len(circuit_out.detectors())
        assert circuit_in.detectors() == circuit_out.detectors()

    def test_detectors_are_more_ordered_when_lexical_flag_set(self, stim_circuit):
        _, _, stim_circuit_out = parse_stim_circuit(
            stim_circuit, trim_circuit=False, lexical_detectors=True
        )
        circuit_in = sp.Circuit.from_stim_circuit(stim_circuit)
        circuit_out = sp.Circuit.from_stim_circuit(stim_circuit_out)
        assert len(circuit_in.detectors()) == len(circuit_out.detectors())

        firsts_out, seconds_out = tee(circuit_out.detectors())
        firsts_in, seconds_in = tee(circuit_in.detectors())
        next(seconds_out), next(seconds_in)
        output_orderings = sum(
            a.coordinate <= b.coordinate for a, b in zip(firsts_out, seconds_out)
        )
        input_orderings = sum(
            a.coordinate <= b.coordinate for a, b in zip(firsts_in, seconds_in)
        )

        assert output_orderings >= input_orderings


def get_untrimmed_example_stim(test_case_id: int, reference_data_dir: Path) -> str:
    data_dir = reference_data_dir / "trim_examples" / "untrimmed"
    circuit_file_path = data_dir / f"TestCase-{test_case_id}.stim"
    with Path.open(circuit_file_path, "r", encoding="utf-8") as circuit_file:
        return circuit_file.read()


def get_trimmed_example_stim(test_case_id: int, reference_data_dir: Path) -> str:
    data_dir = reference_data_dir / "trim_examples" / "trimmed"
    circuit_file_path = data_dir / f"TestCase-{test_case_id}-trimmed.stim"
    with Path.open(circuit_file_path, "r", encoding="utf-8") as circuit_file:
        return circuit_file.read()


def get_expected_nodes_20549():
    nodes = []
    for i in range(39):
        start = 12 + 24 * i
        sub = [*list(range(start, start + 2)), *list(range(start + 4, start + 14))]
        nodes += sub
    return set(nodes)


def get_expected_nodes_20612():
    nodes = []
    for i in range(39):
        start = 20 + 40 * i
        sub = [*list(range(start, start + 4)), *list(range(start + 24, start + 40))]
        nodes += sub
    return set(nodes)


class TestGetIrrelevantNodes:
    @pytest.mark.parametrize(("use_lestim"), [False, True])
    @pytest.mark.parametrize(
        ("test_case_id", "expected_detectors"),
        [
            (19977, set()),  # Full mem rep
            (
                20277,
                {
                    4,
                    6,
                    7,
                    8,
                    12,
                    14,
                    15,
                    16,
                    20,
                    22,
                    23,
                    24,
                    28,
                    30,
                    31,
                    32,
                    36,
                    38,
                    39,
                    40,
                    44,
                    46,
                    47,
                    48,
                    52,
                    54,
                    55,
                    56,
                },
            ),  # Full mem rplanar
            (20549, get_expected_nodes_20549()),  # Full mem rplanar min df-3
            (20612, get_expected_nodes_20612()),  # Full mem unrot planar min df-3
            (20710, set()),  # Half mem rplanar
            (
                20782,
                {
                    6,
                    7,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    20,
                    21,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    34,
                    35,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    48,
                    49,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    62,
                    63,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    76,
                    77,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                },
            ),  # Rectangular mem rplanar
            (
                20875,
                (
                    {
                        0,
                        1,
                        2,
                        3,
                        4,
                        128,
                        129,
                        130,
                        131,
                        9,
                        10,
                        11,
                        12,
                        13,
                        145,
                        146,
                        147,
                        148,
                        149,
                        132,
                        26,
                        27,
                        28,
                        29,
                        30,
                        158,
                        159,
                        160,
                        161,
                        162,
                        43,
                        44,
                        45,
                        46,
                        47,
                        60,
                        61,
                        62,
                        63,
                        64,
                        77,
                        78,
                        79,
                        80,
                        81,
                        94,
                        95,
                        96,
                        97,
                        98,
                        111,
                        112,
                        113,
                        114,
                        115,
                    }
                ),
            ),  # Full stab rplanar
            (21174, {0, 5, 10, 15, 20, 21}),  # Full stab rplanar
            (21367, set()),  # Half stab rplanar
        ],
    )
    def test_get_irrelevant_nodes(
        self,
        reference_data_dir: Path,
        test_case_id: int,
        expected_detectors: set[int],
        use_lestim: bool,
    ):
        circuit_str = get_untrimmed_example_stim(test_case_id, reference_data_dir)
        circuit = StimCircuit(circuit_str) if use_lestim else stim.Circuit(circuit_str)
        assert get_irrelevant_nodes(circuit) == expected_detectors

    # We dont need to test with stim as leakage is unsupported.
    @pytest.mark.parametrize(
        ("test_case_id", "expected_detectors"),
        [
            (19979, set()),
            (
                20279,
                {
                    4,
                    6,
                    7,
                    8,
                    12,
                    14,
                    15,
                    16,
                    20,
                    22,
                    23,
                    24,
                    28,
                    30,
                    31,
                    32,
                    36,
                    38,
                    39,
                    40,
                    44,
                    46,
                    47,
                    48,
                    52,
                    54,
                    55,
                    56,
                },
            ),
            (20548, get_expected_nodes_20549()),
        ],
    )
    def test_get_irrelevant_nodes_leakage(
        self,
        reference_data_dir: Path,
        test_case_id: int,
        expected_detectors: set[int],
    ):
        # If lestim is installed then run the test, else ignore.
        if find_spec("lestim") is not None:
            circuit_str = get_untrimmed_example_stim(test_case_id, reference_data_dir)
            circuit = StimCircuit(circuit_str)
            assert get_irrelevant_nodes(circuit) == expected_detectors


class TestGetTrimmedCircuit:
    @pytest.mark.parametrize(("use_lestim"), [False, True])
    @pytest.mark.parametrize(
        ("test_case_id"),
        [
            (19977),  # Full mem rep
            (20277),  # Full mem rplanar
            (20549),  # Full mem rplanar min df-3
            (20612),  # Full mem unrot planar min df-3
            (20710),  # Half mem rplanar
            (20782),  # Rectangular mem rplanar
            (20875),  # Full stab rplanar
            (21174),  # Full stab rplanar
            (21367),  # Half stab rplanar
        ],
    )
    def test_get_trimmed_circuit(
        self,
        reference_data_dir: Path,
        test_case_id: int,
        use_lestim: bool,
    ):
        untrimmed_circuit_str = get_untrimmed_example_stim(
            test_case_id, reference_data_dir
        )
        trimmed_circuit_str = get_trimmed_example_stim(test_case_id, reference_data_dir)
        if use_lestim:
            circuit = StimCircuit(untrimmed_circuit_str)
        else:
            circuit = stim.Circuit(untrimmed_circuit_str)
        assert str(get_trimmed_circuit(circuit)) == trimmed_circuit_str

    # We anticipate a failure as we need to cast to stim with this function.
    @pytest.mark.parametrize(
        ("test_case_id"),
        [
            (19979),
            (20279),
            (20548),
        ],
    )
    def test_get_trimmed_circuit_leakage_failure(
        self,
        reference_data_dir: Path,
        test_case_id: int,
    ):
        # If lestim is installed then run the test, else ignore.
        if find_spec("lestim") is not None:
            untrimmed_circuit_str = get_untrimmed_example_stim(
                test_case_id, reference_data_dir
            )
            circuit = StimCircuit(untrimmed_circuit_str)
            with pytest.raises(NotImplementedError):
                get_trimmed_circuit(circuit)
