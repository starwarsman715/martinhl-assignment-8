import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

# Constants
RESULT_DIR = "results"
DEFAULT_SAMPLES = 100
DEFAULT_CLUSTER_STD = 0.5

def initialize_results_directory():
    """Ensure results directory exists and is empty"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    # Clear previous results
    for file in os.listdir(RESULT_DIR):
        os.remove(os.path.join(RESULT_DIR, file))

def generate_ellipsoid_clusters(distance, n_samples=DEFAULT_SAMPLES, cluster_std=DEFAULT_CLUSTER_STD):
    """Generate two ellipsoid clusters with specified separation distance"""
    np.random.seed(0)  # For reproducibility
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                 [cluster_std * 0.8, cluster_std]])
    
    # Generate clusters
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2[:, 0] += distance  # Shift along x-axis
    X2[:, 1] += distance  # Shift along y-axis
    
    # Create labels
    y1 = np.zeros(n_samples)
    y2 = np.ones(n_samples)
    
    return np.vstack((X1, X2)), np.hstack((y1, y2))

def calculate_logistic_loss(model, X, y):
    """Calculate the logistic loss (cross-entropy)"""
    y_pred = model.predict_proba(X)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred[:, 1]) + (1 - y) * np.log(1 - y_pred[:, 1]))

def fit_logistic_regression(X, y):
    """Fit logistic regression and return model with parameters"""
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def validate_parameters(start, end, step_num):
    """Validate input parameters"""
    if not (0 < start < end):
        raise ValueError("Start must be positive and less than end")
    if not (2 <= step_num <= 20):
        raise ValueError("Step number must be between 2 and 20")
    if end > 10:
        raise ValueError("End value must not exceed 10")

def do_experiments(start, end, step_num):
    """Run logistic regression experiments with error handling"""
    try:
        # Validate parameters
        validate_parameters(start, end, step_num)
        
        # Initialize results directory
        initialize_results_directory()
        
        # Setup parameters
        shift_distances = np.linspace(start, end, step_num)
        beta0_list, beta1_list, beta2_list = [], [], []
        slope_list, intercept_list, loss_list, margin_widths = [], [], [], []
        sample_data = {}

        # Calculate grid layout
        n_cols = min(3, step_num)
        n_rows = (step_num + n_cols - 1) // n_cols
        
        # Create dataset visualization
        plt.figure(figsize=(20, n_rows * 7))
        
        # Run experiments for each distance
        for i, distance in enumerate(shift_distances, 1):
            # Generate and fit data
            X, y = generate_ellipsoid_clusters(distance=distance)
            model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
            
            # Store parameters
            beta0_list.append(beta0)
            beta1_list.append(beta1)
            beta2_list.append(beta2)
            slope = -beta1 / beta2
            intercept = -beta0 / beta2
            slope_list.append(slope)
            intercept_list.append(intercept)
            loss_list.append(calculate_logistic_loss(model, X, y))

            # Plot dataset
            plt.subplot(n_rows, n_cols, i)
            plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0')
            plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1')
            
            # Setup plot boundaries
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            
            # Plot decision boundary
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), 
                                np.linspace(y_min, y_max, 200))
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            
            plt.plot([x_min, x_max], 
                    [-(beta0 + beta1*x_min)/beta2, -(beta0 + beta1*x_max)/beta2], 
                    'k-', label='Decision Boundary')

            # Plot confidence contours
            contour_levels = [0.7, 0.8, 0.9]
            alphas = [0.05, 0.1, 0.15]
            for level, alpha in zip(contour_levels, alphas):
                class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], 
                                             colors=['red'], alpha=alpha)
                class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], 
                                             colors=['blue'], alpha=alpha)
                if level == 0.7:
                    distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices, 
                                    class_0_contour.collections[0].get_paths()[0].vertices)
                    min_distance = np.min(distances)
                    margin_widths.append(min_distance)

            # Add plot labels and text
            plt.title(f"Shift Distance = {distance:.2f}", fontsize=16)
            plt.xlabel("x1", fontsize=12)
            plt.ylabel("x2", fontsize=12)
            
            equation_text = f"{beta0:.2f} + {beta1:.2f}x1 + {beta2:.2f}x2 = 0\ny = {slope:.2f}x + {intercept:.2f}"
            margin_text = f"Margin Width: {min_distance:.2f}"
            plt.text(x_min + 0.1, y_max - 0.5, equation_text, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            plt.text(x_min + 0.1, y_max - 1.0, margin_text, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))

            if i == 1:
                plt.legend(loc='lower right')

            sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "dataset.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create parameter visualization
        plt.figure(figsize=(18, 15))
        
        # Plot all parameters
        param_plots = [
            (beta0_list, "Beta0", "b-"),
            (beta1_list, "Beta1 (x1 coefficient)", "r-"),
            (beta2_list, "Beta2 (x2 coefficient)", "g-"),
            (slope_list, "Slope (Beta1/Beta2)", "m-"),
            ([-b0/b2 for b0, b2 in zip(beta0_list, beta2_list)], "Intercept Ratio (Beta0/Beta2)", "c-"),
            (loss_list, "Logistic Loss", "y-"),
            (margin_widths, "Margin Width", "k-")
        ]
        
        for i, (data, title, style) in enumerate(param_plots, 1):
            plt.subplot(3, 3, i)
            plt.plot(shift_distances, data, style, marker='o')
            plt.title(title)
            plt.xlabel("Shift Distance")
            plt.ylabel(title.split()[0])
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "parameters_vs_shift_distance.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "status": "success",
            "message": "Experiments completed successfully",
            "dataset_img": "results/dataset.png",
            "parameters_img": "results/parameters_vs_shift_distance.png"
        }

    except Exception as e:
        plt.close('all')  # Clean up any open plots
        return {
            "status": "error",
            "message": str(e),
            "dataset_img": None,
            "parameters_img": None
        }

if __name__ == "__main__":
    # Test with default values
    result = do_experiments(0.25, 2.0, 8)
    print(result)