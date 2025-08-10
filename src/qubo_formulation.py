# src/qubo_formulation.py
"""
QUBO Formulation for Constrained Portfolio Optimization
Converts a portfolio selection problem into a QUBO with penalty terms.
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from typing import List, Tuple, Optional

class PortfolioQUBOBuilder:
    """
    Builds a QUBO for binary portfolio optimization with:
    - Risk minimization
    - Return maximization
    - Cardinality constraint (exactly K assets)
    - Target return constraint
    - Sparsity and cost penalties
    """
    
    def __init__(
        self,
        expected_returns: List[float],
        cov_matrix: np.ndarray,
        K: int,
        target_return: float,
        risk_aversion: float = 0.5,
        sparsity: float = 0.01,
        return_penalty_weight: float = 10.0,
        cardinality_penalty_weight: float = 10.0
    ):
        """
        Args:
            expected_returns: List of expected returns for each asset
            cov_matrix: Covariance matrix (N x N)
            K: Number of assets to select (cardinality)
            target_return: Minimum acceptable portfolio return
            risk_aversion: Weight for return in objective
            sparsity: L1 penalty for reducing small weights
            return_penalty_weight: Penalty strength for return constraint
            cardinality_penalty_weight: Penalty strength for K constraint
        """
        self.expected_returns = np.array(expected_returns)
        self.cov_matrix = np.array(cov_matrix)
        self.N = len(expected_returns)
        self.K = K
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.sparsity = sparsity
        self.return_penalty_weight = return_penalty_weight
        self.cardinality_penalty_weight = cardinality_penalty_weight

    def build(self) -> QuadraticProgram:
        """
        Build the full constrained quadratic program.
        Returns a QuadraticProgram object ready for conversion to QUBO.
        """
        qp = QuadraticProgram()

        # Add binary variables: x_i = 1 if asset i is selected
        for i in range(self.N):
            qp.binary_var(name=f'x_{i}')

        # Objective: minimize risk - maximize return + sparsity
        quadratic = self.cov_matrix
        linear = -self.risk_aversion * self.expected_returns + self.sparsity

        qp.minimize(quadratic=quadratic, linear=linear)

        # Constraint 1: Exactly K assets selected
        qp.linear_constraint(
            linear=[1] * self.N,
            sense='==',
            rhs=self.K,
            name='cardinality'
        )

        # Constraint 2: Portfolio return >= target_return
        qp.linear_constraint(
            linear=self.expected_returns.tolist(),
            sense='>=',
            rhs=self.target_return,
            name='min_return'
        )

        return qp

    def build_qubo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build QUBO matrix Q and vector c such that: cost = x^T Q x + c^T x
        Returns:
            Q (ndarray): Quadratic coefficients (N x N)
            c (ndarray): Linear coefficients (N,)
        """
        from qiskit_optimization.converters import QuadraticProgramToQubo

        qp = self.build()
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        Q, c, _ = qubo.to_quadratic_form()

        return Q, c

    def build_ising(self):
        """
        Convert to Ising Hamiltonian for quantum solvers.
        Returns: operator, offset
        """
        from qiskit_optimization.converters import QuadraticProgramToQubo

        qp = self.build()
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        return qubo.to_ising()
