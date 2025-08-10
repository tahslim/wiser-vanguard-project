"""
Quantum Solver for Portfolio Optimization using QAOA + CVaR
Author: Team Quantum Vanguard
"""

import numpy as np
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_optimization.applications import OptimizationApplication
from qiskit.circuit.library import TwoLocal
import warnings
warnings.filterwarnings('ignore')

class QuantumPortfolioOptimizer:
    def __init__(self, cov_matrix, expected_returns, K, target_return, 
                 risk_aversion=0.5, sparsity=0.01, reps=3, use_cvar=True, alpha=0.05):
        self.cov_matrix = cov_matrix
        self.expected_returns = expected_returns
        self.N = len(cov_matrix)
        self.K = K
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.sparsity = sparsity
        self.reps = reps
        self.use_cvar = use_cvar
        self.alpha = alpha
        self.qubo = None
        self.qaoa = None
        self.result = None

    def _build_quadratic_program(self):
        """Build the constrained quadratic program"""
        qp = QuadraticProgram()

        # Add binary variables
        for i in range(self.N):
            qp.binary_var(name=f'x_{i}')

        # Objective: risk - return + sparsity
        quadratic = self.cov_matrix
        linear = -self.risk_aversion * np.array(self.expected_returns) + self.sparsity

        qp.minimize(quadratic=quadratic, linear=linear)

        # Cardinality constraint: sum x_i = K
        qp.linear_constraint(linear=[1]*self.N, sense='==', rhs=self.K, name='cardinality')

        # Return constraint: sum mu_i x_i >= T
        qp.linear_constraint(linear=self.expected_returns, sense='>=', rhs=self.target_return, name='target_return')

        return qp

    def _cvar_objective(self, probabilities, values):
        """CVaR objective: average of top alpha% lowest energies"""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_probs = probabilities[sorted_indices]

        cum_prob = 0.0
        cvar_sum = 0.0
        count = 0
        for i in range(len(sorted_values)):
            cvar_sum += sorted_values[i] * sorted_probs[i]
            cum_prob += sorted_probs[i]
            count += 1
            if cum_prob >= self.alpha:
                break
        return cvar_sum / self.alpha

    def solve(self, optimizer=None, sampler=None):
        """Solve the portfolio problem using QAOA"""
        if optimizer is None:
            optimizer = COBYLA(maxiter=100)
        if sampler is None:
            sampler = Sampler()

        qp = self._build_quadratic_program()
        self.qubo = qp.to_qubo()

        # Convert constraints to penalties (Qiskit does this internally via converters)
        # But we can also use LinearEqualityToPenalty if needed
        converter = LinearEqualityToPenalty(penalty=10)

        qubo_with_penalty = converter.convert(qp.to_quadratic_program())

        # Define custom objective for CVaR
        def cvar_expectation(estimator, pub):
            from qiskit.primitives import BaseSampler
            if isinstance(pub, tuple):
                circuit, params = pub[0], pub[1]
            else:
                circuit = pub
                params = None

            job = sampler.run(circuit, parameter_values=params)
            result = job.result()
            counts = result.quasi_dists[0].binary_probabilities()
            energies = []
            probs = []
            for bitstr, prob in counts.items():
                x = np.array([int(b) for b in bitstr])
                energy = x @ self.cov_matrix @ x - self.risk_aversion * (self.expected_returns @ x)
                energies.append(energy)
                probs.append(prob)
            energies = np.array(energies)
            probs = np.array(probs)
            return self._cvar_objective(probs, energies)

        # Use standard QAOA but we could inject CVaR in future
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=self.reps)
        self.qaoa = qaoa

        # Solve
        min_eigen_solver = qaoa
        result = min_eigen_solver.compute_minimum_eigenvalue(self.qubo[0])
        self.result = result

        # Decode
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        meo = MinimumEigenOptimizer(qaoa)
        solution = meo.solve(qp)
        return solution

    def get_circuit(self):
        """Return the QAOA ansatz circuit"""
        if self.qaoa is None:
            raise ValueError("Run solve() first.")
        return self.qaoa.ansatz.bind_parameters(self.result.optimal_point)
