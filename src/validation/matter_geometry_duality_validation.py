"""
Matter-Geometry Duality Control Parameter Validation
Addresses Severity 91 UQ concern with comprehensive validation framework
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, hessian
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigvals, solve_lyapunov
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings

@dataclass
class DualityControlParameters:
    """Container for matter-geometry duality control parameters"""
    coupling_strength: float  # α ∈ [0, 1]
    feedback_gain: float     # K > 0
    stability_margin: float  # σ ∈ [0.1, 0.9]
    response_time: float     # τ > 0
    nonlinearity_factor: float  # β ∈ [0, 0.5]

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    lyapunov_stable: bool
    controllable: bool
    observable: bool
    robust_stable: bool
    parameter_sensitivity: Dict[str, float]
    stability_margins: Dict[str, float]

class MatterGeometryDualityValidator:
    """
    Comprehensive validator for matter-geometry duality control parameters
    Ensures stable bidirectional coupling between matter and spacetime geometry
    """
    
    def __init__(self):
        self.c = 299792458  # Speed of light
        self.G = 6.67430e-11  # Gravitational constant
        self.hbar = 1.054571817e-34  # Reduced Planck constant
        
        # Fundamental coupling bounds from general relativity + quantum mechanics
        self.fundamental_bounds = {
            'coupling_strength': (1e-6, 0.9),  # Weak to strong coupling
            'feedback_gain': (1e-3, 1e3),      # Stable feedback range
            'stability_margin': (0.05, 0.95),   # Safety margins
            'response_time': (1e-12, 1e-3),     # Femtosecond to millisecond
            'nonlinearity_factor': (0, 0.3)     # Linear to mildly nonlinear
        }
    
    def validate_parameter_bounds(self, params: DualityControlParameters) -> Dict[str, bool]:
        """Validate that parameters are within fundamental physical bounds"""
        validation = {}
        
        param_dict = {
            'coupling_strength': params.coupling_strength,
            'feedback_gain': params.feedback_gain,
            'stability_margin': params.stability_margin,
            'response_time': params.response_time,
            'nonlinearity_factor': params.nonlinearity_factor
        }
        
        for param_name, value in param_dict.items():
            lower, upper = self.fundamental_bounds[param_name]
            validation[f'{param_name}_in_bounds'] = lower <= value <= upper
            validation[f'{param_name}_physical'] = self._check_physical_consistency(param_name, value)
        
        return validation
    
    def _check_physical_consistency(self, param_name: str, value: float) -> bool:
        """Check physical consistency of parameters"""
        if param_name == 'coupling_strength':
            # Must not violate equivalence principle
            return value < 0.9  # Strong but not complete coupling
        
        elif param_name == 'feedback_gain':
            # Must not cause runaway instabilities
            return 1e-3 <= value <= 1e3
        
        elif param_name == 'response_time':
            # Must be slower than Planck time, faster than relaxation
            planck_time = np.sqrt(self.hbar * self.G / self.c**5)
            return value > planck_time * 1e6  # 6 orders above Planck time
        
        elif param_name == 'nonlinearity_factor':
            # Must preserve perturbative validity
            return value < 0.5
        
        else:
            return True
    
    def analyze_lyapunov_stability(self, params: DualityControlParameters) -> Dict[str, float]:
        """Analyze Lyapunov stability of the duality control system"""
        
        # Construct linearized system matrix A for dx/dt = Ax + Bu
        # State vector: x = [matter_density, geometry_curvature, coupling_field, momentum]
        
        α = params.coupling_strength
        K = params.feedback_gain
        τ = params.response_time
        β = params.nonlinearity_factor
        
        # System matrix incorporating matter-geometry coupling
        A = np.array([
            [-1/τ,     α*K,      α,       0     ],  # Matter density evolution
            [α*K,      -2/τ,     α*K,     0     ],  # Geometry curvature evolution
            [α,        α*K,      -3/τ,    β     ],  # Coupling field dynamics
            [0,        0,        β,       -1/(2*τ)]  # Momentum conservation
        ])
        
        # Compute eigenvalues for stability analysis
        eigenvalues = eigvals(A)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Lyapunov stability: all eigenvalues have negative real parts
        lyapunov_stable = np.all(real_parts < -1e-6)
        
        # Stability margins
        min_real_part = np.min(real_parts)
        max_real_part = np.max(real_parts)
        stability_margin = abs(min_real_part) if lyapunov_stable else -max_real_part
        
        # Oscillation analysis
        oscillatory_modes = np.sum(np.abs(imag_parts) > 1e-6)
        max_frequency = np.max(np.abs(imag_parts)) / (2 * np.pi)
        
        # Condition number for numerical stability
        condition_number = np.linalg.cond(A)
        
        return {
            'lyapunov_stable': lyapunov_stable,
            'stability_margin': stability_margin,
            'min_real_eigenvalue': min_real_part,
            'max_real_eigenvalue': max_real_part,
            'oscillatory_modes': oscillatory_modes,
            'max_frequency_hz': max_frequency,
            'condition_number': condition_number,
            'eigenvalues': eigenvalues.tolist()
        }
    
    def assess_controllability_observability(self, params: DualityControlParameters) -> Dict[str, bool]:
        """Assess controllability and observability of the duality system"""
        
        α = params.coupling_strength
        K = params.feedback_gain
        τ = params.response_time
        β = params.nonlinearity_factor
        
        # System matrices
        A = np.array([
            [-1/τ,     α*K,      α,       0     ],
            [α*K,      -2/τ,     α*K,     0     ],
            [α,        α*K,      -3/τ,    β     ],
            [0,        0,        β,       -1/(2*τ)]
        ])
        
        # Control matrix B (can control matter density and coupling field)
        B = np.array([
            [1, 0],    # Control matter density
            [0, 0],    # Cannot directly control geometry
            [0, 1],    # Control coupling field
            [0, 0]     # Cannot directly control momentum
        ])
        
        # Observation matrix C (can observe all states)
        C = np.eye(4)
        
        # Controllability matrix: [B, AB, A²B, A³B]
        controllability_matrix = np.hstack([
            B,
            A @ B,
            A @ A @ B,
            A @ A @ A @ B
        ])
        
        # Observability matrix: [C; CA; CA²; CA³]
        observability_matrix = np.vstack([
            C,
            C @ A,
            C @ A @ A,
            C @ A @ A @ A
        ])
        
        # Check ranks
        n_states = A.shape[0]
        controllable = np.linalg.matrix_rank(controllability_matrix) == n_states
        observable = np.linalg.matrix_rank(observability_matrix) == n_states
        
        # Gramian-based analysis for numerical robustness
        try:
            # Solve Lyapunov equations for Gramians
            Wc = solve_lyapunov(A, -B @ B.T)  # Controllability Gramian
            Wo = solve_lyapunov(A.T, -C.T @ C)  # Observability Gramian
            
            controllability_gramian_min_eigenvalue = np.min(eigvals(Wc))
            observability_gramian_min_eigenvalue = np.min(eigvals(Wo))
            
            gramian_controllable = controllability_gramian_min_eigenvalue > 1e-12
            gramian_observable = observability_gramian_min_eigenvalue > 1e-12
            
        except np.linalg.LinAlgError:
            gramian_controllable = False
            gramian_observable = False
            controllability_gramian_min_eigenvalue = 0
            observability_gramian_min_eigenvalue = 0
        
        return {
            'controllable': controllable and gramian_controllable,
            'observable': observable and gramian_observable,
            'controllability_rank': np.linalg.matrix_rank(controllability_matrix),
            'observability_rank': np.linalg.matrix_rank(observability_matrix),
            'controllability_gramian_min_eig': controllability_gramian_min_eigenvalue,
            'observability_gramian_min_eig': observability_gramian_min_eigenvalue
        }
    
    def analyze_parameter_sensitivity(self, params: DualityControlParameters) -> Dict[str, float]:
        """Analyze sensitivity of stability to parameter variations"""
        
        base_params = params
        perturbation = 0.01  # 1% perturbation
        
        # Compute baseline stability margin
        baseline_stability = self.analyze_lyapunov_stability(base_params)
        baseline_margin = baseline_stability['stability_margin']
        
        sensitivities = {}
        
        # Test sensitivity to each parameter
        param_names = ['coupling_strength', 'feedback_gain', 'stability_margin', 
                      'response_time', 'nonlinearity_factor']
        
        for param_name in param_names:
            # Create perturbed parameters
            perturbed_params = DualityControlParameters(
                coupling_strength=base_params.coupling_strength,
                feedback_gain=base_params.feedback_gain,
                stability_margin=base_params.stability_margin,
                response_time=base_params.response_time,
                nonlinearity_factor=base_params.nonlinearity_factor
            )
            
            # Perturb specific parameter
            original_value = getattr(perturbed_params, param_name)
            setattr(perturbed_params, param_name, original_value * (1 + perturbation))
            
            # Compute perturbed stability
            perturbed_stability = self.analyze_lyapunov_stability(perturbed_params)
            perturbed_margin = perturbed_stability['stability_margin']
            
            # Compute sensitivity: (Δmargin/margin) / (Δparam/param)
            relative_margin_change = (perturbed_margin - baseline_margin) / baseline_margin
            relative_param_change = perturbation
            sensitivity = relative_margin_change / relative_param_change
            
            sensitivities[param_name] = abs(sensitivity)
        
        return sensitivities
    
    def validate_robustness(self, params: DualityControlParameters, 
                          uncertainty_bounds: Dict[str, float]) -> Dict[str, bool]:
        """Validate robustness against parameter uncertainties"""
        
        # Monte Carlo analysis for robustness
        n_samples = 1000
        stable_count = 0
        
        for _ in range(n_samples):
            # Generate random parameter variations within uncertainty bounds
            perturbed_params = DualityControlParameters(
                coupling_strength=params.coupling_strength * (1 + np.random.uniform(
                    -uncertainty_bounds.get('coupling_strength', 0.1),
                    uncertainty_bounds.get('coupling_strength', 0.1))),
                feedback_gain=params.feedback_gain * (1 + np.random.uniform(
                    -uncertainty_bounds.get('feedback_gain', 0.1),
                    uncertainty_bounds.get('feedback_gain', 0.1))),
                stability_margin=params.stability_margin * (1 + np.random.uniform(
                    -uncertainty_bounds.get('stability_margin', 0.05),
                    uncertainty_bounds.get('stability_margin', 0.05))),
                response_time=params.response_time * (1 + np.random.uniform(
                    -uncertainty_bounds.get('response_time', 0.1),
                    uncertainty_bounds.get('response_time', 0.1))),
                nonlinearity_factor=params.nonlinearity_factor * (1 + np.random.uniform(
                    -uncertainty_bounds.get('nonlinearity_factor', 0.1),
                    uncertainty_bounds.get('nonlinearity_factor', 0.1)))
            )
            
            # Check if perturbed system is stable
            stability_result = self.analyze_lyapunov_stability(perturbed_params)
            if stability_result['lyapunov_stable']:
                stable_count += 1
        
        robustness_percentage = stable_count / n_samples
        
        return {
            'robust_stable': robustness_percentage > 0.95,  # 95% robustness threshold
            'robustness_percentage': robustness_percentage,
            'stable_samples': stable_count,
            'total_samples': n_samples
        }
    
    def optimize_parameters(self, constraints: Dict[str, Tuple[float, float]]) -> DualityControlParameters:
        """Optimize parameters for maximum stability margin while respecting constraints"""
        
        def objective(x):
            # Unpack parameters
            coupling_strength, feedback_gain, stability_margin, response_time, nonlinearity_factor = x
            
            params = DualityControlParameters(
                coupling_strength=coupling_strength,
                feedback_gain=feedback_gain,
                stability_margin=stability_margin,
                response_time=response_time,
                nonlinearity_factor=nonlinearity_factor
            )
            
            # Compute stability margin (maximize this)
            stability_result = self.analyze_lyapunov_stability(params)
            
            if not stability_result['lyapunov_stable']:
                return 1e6  # Penalty for unstable systems
            
            # Return negative stability margin (for minimization)
            return -stability_result['stability_margin']
        
        # Set up bounds
        bounds = [
            constraints.get('coupling_strength', self.fundamental_bounds['coupling_strength']),
            constraints.get('feedback_gain', self.fundamental_bounds['feedback_gain']),
            constraints.get('stability_margin', self.fundamental_bounds['stability_margin']),
            constraints.get('response_time', self.fundamental_bounds['response_time']),
            constraints.get('nonlinearity_factor', self.fundamental_bounds['nonlinearity_factor'])
        ]
        
        # Optimize using differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,
            tol=1e-12,
            workers=1
        )
        
        optimal_params = DualityControlParameters(
            coupling_strength=result.x[0],
            feedback_gain=result.x[1],
            stability_margin=result.x[2],
            response_time=result.x[3],
            nonlinearity_factor=result.x[4]
        )
        
        return optimal_params
    
    def comprehensive_validation(self, params: DualityControlParameters) -> ValidationMetrics:
        """Run comprehensive validation of all aspects"""
        
        # Parameter bounds validation
        bounds_validation = self.validate_parameter_bounds(params)
        
        # Stability analysis
        stability_analysis = self.analyze_lyapunov_stability(params)
        
        # Controllability/observability
        control_analysis = self.assess_controllability_observability(params)
        
        # Parameter sensitivity
        sensitivity_analysis = self.analyze_parameter_sensitivity(params)
        
        # Robustness analysis with 10% uncertainty bounds
        uncertainty_bounds = {param: 0.1 for param in ['coupling_strength', 'feedback_gain', 
                             'stability_margin', 'response_time', 'nonlinearity_factor']}
        robustness_analysis = self.validate_robustness(params, uncertainty_bounds)
        
        return ValidationMetrics(
            lyapunov_stable=stability_analysis['lyapunov_stable'],
            controllable=control_analysis['controllable'],
            observable=control_analysis['observable'],
            robust_stable=robustness_analysis['robust_stable'],
            parameter_sensitivity=sensitivity_analysis,
            stability_margins={
                'lyapunov_margin': stability_analysis['stability_margin'],
                'condition_number': stability_analysis['condition_number'],
                'robustness_percentage': robustness_analysis['robustness_percentage']
            }
        )

def validate_matter_geometry_duality():
    """Main validation function for matter-geometry duality control"""
    
    print("🔧 MATTER-GEOMETRY DUALITY CONTROL VALIDATION")
    print("=" * 55)
    
    validator = MatterGeometryDualityValidator()
    
    # Test with realistic parameters
    test_params = DualityControlParameters(
        coupling_strength=0.15,      # Moderate coupling
        feedback_gain=2.5,          # Stable feedback
        stability_margin=0.3,       # 30% safety margin
        response_time=1e-6,         # Microsecond response
        nonlinearity_factor=0.05    # Weak nonlinearity
    )
    
    print(f"Testing parameters:")
    print(f"  Coupling strength: {test_params.coupling_strength}")
    print(f"  Feedback gain: {test_params.feedback_gain}")
    print(f"  Stability margin: {test_params.stability_margin}")
    print(f"  Response time: {test_params.response_time:.2e} s")
    print(f"  Nonlinearity factor: {test_params.nonlinearity_factor}")
    
    # Run comprehensive validation
    results = validator.comprehensive_validation(test_params)
    
    print("\n📊 VALIDATION RESULTS:")
    print(f"  Lyapunov stable: {'✅ YES' if results.lyapunov_stable else '❌ NO'}")
    print(f"  Controllable: {'✅ YES' if results.controllable else '❌ NO'}")
    print(f"  Observable: {'✅ YES' if results.observable else '❌ NO'}")
    print(f"  Robust stable: {'✅ YES' if results.robust_stable else '❌ NO'}")
    
    print(f"\n📈 STABILITY MARGINS:")
    for key, value in results.stability_margins.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n🎯 PARAMETER SENSITIVITY:")
    for param, sensitivity in results.parameter_sensitivity.items():
        sensitivity_level = "LOW" if sensitivity < 1 else "MEDIUM" if sensitivity < 10 else "HIGH"
        print(f"  {param}: {sensitivity:.2f} ({sensitivity_level})")
    
    # Optimize parameters for better performance
    print(f"\n🚀 OPTIMIZING PARAMETERS...")
    constraints = {
        'coupling_strength': (0.05, 0.4),
        'feedback_gain': (0.5, 10.0),
        'response_time': (1e-9, 1e-4)
    }
    
    optimal_params = validator.optimize_parameters(constraints)
    optimal_results = validator.comprehensive_validation(optimal_params)
    
    print(f"\n🎖️ OPTIMIZED PARAMETERS:")
    print(f"  Coupling strength: {optimal_params.coupling_strength:.3f}")
    print(f"  Feedback gain: {optimal_params.feedback_gain:.3f}")
    print(f"  Stability margin: {optimal_params.stability_margin:.3f}")
    print(f"  Response time: {optimal_params.response_time:.2e} s")
    print(f"  Nonlinearity factor: {optimal_params.nonlinearity_factor:.3f}")
    
    print(f"\n✨ OPTIMIZED PERFORMANCE:")
    print(f"  Stability margin: {optimal_results.stability_margins['lyapunov_margin']:.4f}")
    print(f"  Robustness: {optimal_results.stability_margins['robustness_percentage']:.1%}")
    
    overall_success = (
        optimal_results.lyapunov_stable and
        optimal_results.controllable and
        optimal_results.observable and
        optimal_results.robust_stable
    )
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if overall_success:
        print("  ✅ MATTER-GEOMETRY DUALITY CONTROL VALIDATED")
        print("  🚀 PARAMETERS OPTIMIZED FOR STABLE OPERATION")
    else:
        print("  ⚠️ SOME VALIDATION CRITERIA NOT MET")
        print("  🔧 RECOMMEND PARAMETER ADJUSTMENT")
    
    return optimal_params, optimal_results

if __name__ == "__main__":
    validate_matter_geometry_duality()
