import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.solvers import LPassignment, MIPassignment, forbiddentropsubsetDPassignment


@pytest.mark.parametrize(
    "costs, threshold",
    [
        ([[1, 2], [999, 999]], 999),  # A row has no allowed entries.
        ([[1, 999], [2, 999]], 999),  # Both rows need the same column.
        ([[5]], 5),  # Entries equal to the custom threshold are forbidden.
    ],
)
def test_sparse_infeasible_assignment_terminates(costs, threshold):
    # Isolate reconstruction so a regression cannot hang the test runner.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import math; import numpy as np; "
            "from src.solvers import forbiddentropsubsetDPassignment; "
            f"result = forbiddentropsubsetDPassignment(np.array({costs!r}), M={threshold}); "
            "assert result == [math.inf, {}], result",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_sparse_feasible_assignment_with_infeasible_subproblem():
    costs = np.array([[1, 999], [2, 3]])
    assert forbiddentropsubsetDPassignment(costs) == [4, {0: 0, 1: 1}]


@pytest.mark.parametrize("solver", [LPassignment, MIPassignment])
@pytest.mark.parametrize(
    "costs, expected_cost",
    [
        ([[0.25, 4.0], [3.0, 0.5]], 0.75),
        ([[-1.25, 4.0], [3.0, -0.5]], -1.75),
    ],
)
def test_fractional_objective_and_assignment(solver, costs, expected_cost):
    costs = np.array(costs)
    cost, indices = solver(costs)
    assert cost == pytest.approx(expected_cost)
    np.testing.assert_array_equal(indices, [[0, 0], [1, 1]])
    assert cost == pytest.approx(sum(costs[i, j] for i, j in indices))
