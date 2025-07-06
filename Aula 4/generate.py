import numpy as np

def generate_synthetic_data(n_samples=500, data_type='random', random_state=None):
    """Generate synthetic data with either:
    - 'random': Complex multi-modal distributions with overlapping classes
    - 'xor': Classic XOR pattern
    
    Parameters:
    - n_samples: total number of samples
    - data_type: 'random' for complex clusters or 'xor' for XOR pattern
    - random_state: seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if data_type == 'random':
        # Create a more challenging distribution with:
        # - Multiple clusters per class
        # - Overlapping distributions
        # - Different variances
        # - Non-linear decision boundaries
        
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples)
        
        # Class 1 (-1) will have:
        # 1. A dense cluster
        # 2. A ring-shaped distribution
        # 3. A linear cluster
        
        # Class 2 (+1) will have:
        # 1. Two dense clusters
        # 2. A spiral pattern
        
        # Assign samples to different distributions
        dist_probs = [0.3, 0.3, 0.2, 0.1, 0.1]  # Probabilities for each distribution
        dist_choices = np.random.choice(5, size=n_samples, p=dist_probs)
        
        for i in range(n_samples):
            if dist_choices[i] == 0:  # Class -1 dense cluster
                X[i] = np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]])
                y[i] = -1
            elif dist_choices[i] == 1:  # Class -1 ring
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.normal(1.5, 0.1)
                X[i, 0] = radius * np.cos(angle)
                X[i, 1] = radius * np.sin(angle)
                y[i] = -1
            elif dist_choices[i] == 2:  # Class -1 linear
                x = np.random.uniform(-2, 2)
                X[i, 0] = x
                X[i, 1] = 0.5 * x + np.random.normal(0, 0.1)
                y[i] = -1
            elif dist_choices[i] == 3:  # Class +1 cluster 1
                X[i] = np.random.multivariate_normal([1, 1], [[0.2, 0.1], [0.1, 0.2]])
                y[i] = 1
            elif dist_choices[i] == 4:  # Class +1 cluster 2 with spiral
                t = np.random.uniform(0, 2*np.pi)
                r = t / (2*np.pi)
                X[i, 0] = 0.5 + r * np.cos(t) + np.random.normal(0, 0.05)
                X[i, 1] = 0.5 + r * np.sin(t) + np.random.normal(0, 0.05)
                y[i] = 1
        
        # Add some noise points
        n_noise = n_samples // 20
        X_noise = np.random.uniform(-2.5, 2.5, size=(n_noise, 2))
        y_noise = np.random.choice([-1, 1], size=n_noise)
        X = np.vstack([X, X_noise])
        y = np.concatenate([y, y_noise])
        
    elif data_type == 'xor':
        # XOR-like pattern with more complexity
        n_samples_per_quadrant = n_samples // 4
        
        # Generate with different variances and some rotation
        theta = np.pi/8  # Slight rotation angle
        
        # Quadrant 1: positive class
        cov = [[0.2, 0.15], [0.15, 0.2]]
        x1, x2 = np.random.multivariate_normal([1, 1], cov, n_samples_per_quadrant).T
        X1 = np.column_stack((x1*np.cos(theta) - x2*np.sin(theta), 
                             x1*np.sin(theta) + x2*np.cos(theta)))
        y1 = np.ones(n_samples_per_quadrant)
        
        # Quadrant 2: negative class
        cov = [[0.25, -0.1], [-0.1, 0.25]]
        x1, x2 = np.random.multivariate_normal([-1, 1], cov, n_samples_per_quadrant).T
        X2 = np.column_stack((x1*np.cos(theta) - x2*np.sin(theta), 
                             x1*np.sin(theta) + x2*np.cos(theta)))
        y2 = -np.ones(n_samples_per_quadrant)
        
        # Quadrant 3: positive class
        cov = [[0.3, 0], [0, 0.1]]
        x1, x2 = np.random.multivariate_normal([-1, -1], cov, n_samples_per_quadrant).T
        X3 = np.column_stack((x1*np.cos(theta) - x2*np.sin(theta), 
                             x1*np.sin(theta) + x2*np.cos(theta)))
        y3 = np.ones(n_samples_per_quadrant)
        
        # Quadrant 4: negative class
        cov = [[0.1, 0], [0, 0.3]]
        x1, x2 = np.random.multivariate_normal([1, -1], cov, n_samples_per_quadrant).T
        X4 = np.column_stack((x1*np.cos(theta) - x2*np.sin(theta), 
                             x1*np.sin(theta) + x2*np.cos(theta)))
        y4 = -np.ones(n_samples_per_quadrant)
        
        # Combine and shuffle
        X = np.vstack([X1, X2, X3, X4])
        y = np.hstack([y1, y2, y3, y4])
        indices = np.random.permutation(len(X))
        X = X[indices][:n_samples]
        y = y[indices][:n_samples]
        
    else:
        raise ValueError("data_type must be either 'random' or 'xor'")
    
    return X, y