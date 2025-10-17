% =========================================================================
% IE 317 - Nonlinear Optimization
% Lecture 4: Unconstrained Nonlinear Optimization in Many Variables
% MATLAB Examples and Implementations
% =========================================================================

clear all; close all; clc;

%% Example 1: Gradient and Stationary Points
% Example function: f(x) = x1^2 - x2^2

fprintf('=== Example 1: Finding Stationary Points ===\n\n');

% Define symbolic variables
syms x1 x2;

% Define function
f = x1^2 - x2^2;

fprintf('Function: f(x1, x2) = x1^2 - x2^2\n\n');

% Compute gradient
grad_f = [diff(f, x1); diff(f, x2)];
fprintf('Gradient:\n');
disp(grad_f);

% Find stationary points
stationary_eqns = grad_f == [0; 0];
stationary_point = solve(stationary_eqns, [x1, x2]);

fprintf('Stationary point: [%d, %d]\n\n', ...
    double(stationary_point.x1), double(stationary_point.x2));

% Compute Hessian
hessian_f = [diff(grad_f(1), x1), diff(grad_f(1), x2);
             diff(grad_f(2), x1), diff(grad_f(2), x2)];

fprintf('Hessian matrix:\n');
disp(hessian_f);

% Evaluate Hessian at stationary point
H = double(subs(hessian_f, [x1, x2], [0, 0]));
fprintf('Hessian at stationary point:\n');
disp(H);

% Check definiteness
fprintf('Checking definiteness:\n');
fprintf('1st LPM = %.2f\n', H(1,1));
fprintf('2nd LPM = %.2f\n', det(H));

if det(H) < 0
    fprintf('=> Hessian is INDEFINITE (saddle point)\n\n');
elseif all(eig(H) > 0)
    fprintf('=> Hessian is POSITIVE DEFINITE (local minimum)\n\n');
elseif all(eig(H) < 0)
    fprintf('=> Hessian is NEGATIVE DEFINITE (local maximum)\n\n');
else
    fprintf('=> Inconclusive\n\n');
end

% Visualize the saddle point
figure('Name', 'Example 1: Saddle Point Visualization', 'NumberTitle', 'off');
[X1, X2] = meshgrid(-3:0.1:3, -3:0.1:3);
F = X1.^2 - X2.^2;

subplot(1,2,1);
surf(X1, X2, F, 'EdgeColor', 'none');
hold on;
plot3(0, 0, 0, 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
title('3D View: f(x_1, x_2) = x_1^2 - x_2^2', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
colormap('jet');
view(45, 30);
lighting gouraud;
camlight;

subplot(1,2,2);
contour(X1, X2, F, 30, 'LineWidth', 1.5);
hold on;
plot(0, 0, 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
title('Contour Plot: Saddle Point at Origin', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2');
colorbar;
grid on;
axis equal;

pause(2);

%% Example 2: Checking Definiteness of Matrices

fprintf('=== Example 2: Checking Matrix Definiteness ===\n\n');

% Test matrices
H1 = [18, -3; -3, 2];
H2 = [2, 0; 0, -2];
H3 = [3, 2; 1, 0];

matrices = {H1, H2, H3};
names = {'H1 (from homework function)', 'H2 (saddle point)', 'H3 (general)'};

for i = 1:length(matrices)
    H = matrices{i};
    fprintf('Matrix %s:\n', names{i});
    disp(H);
    
    % Compute LPMs
    lpm1 = H(1,1);
    lpm2 = det(H);
    
    fprintf('1st LPM = %.2f\n', lpm1);
    fprintf('2nd LPM = %.2f\n', lpm2);
    
    % Compute eigenvalues
    eigenvals = eig(H);
    fprintf('Eigenvalues: %.4f, %.4f\n', eigenvals(1), eigenvals(2));
    
    % Check definiteness
    if all(eigenvals > 0)
        fprintf('=> POSITIVE DEFINITE\n');
    elseif all(eigenvals >= 0)
        fprintf('=> POSITIVE SEMI-DEFINITE\n');
    elseif all(eigenvals < 0)
        fprintf('=> NEGATIVE DEFINITE\n');
    elseif all(eigenvals <= 0)
        fprintf('=> NEGATIVE SEMI-DEFINITE\n');
    else
        fprintf('=> INDEFINITE\n');
    end
    fprintf('\n');
end

pause(2);

%% Example 3: Bisection Method for Line Search

fprintf('=== Example 3: Bisection Method Implementation ===\n\n');

% Define a univariate function for line search
f_line = @(alpha) (alpha - 2)^2 + 1;

% Bisection parameters
a = 0;
b = 5;
epsilon = 0.01;

fprintf('Finding minimum of f(alpha) = (alpha - 2)^2 + 1\n');
fprintf('Initial interval: [%.2f, %.2f]\n', a, b);
fprintf('Tolerance: %.4f\n\n', epsilon);

% Implement bisection
iteration = 0;
while (b - a) > epsilon
    iteration = iteration + 1;
    mid = (a + b) / 2;
    
    % Evaluate at quarter points
    x1 = a + (b - a) / 4;
    x2 = b - (b - a) / 4;
    
    fprintf('Iteration %d: [%.4f, %.4f], midpoint = %.4f\n', ...
        iteration, a, b, mid);
    
    if f_line(x1) < f_line(x2)
        b = x2;
    else
        a = x1;
    end
end

alpha_opt = (a + b) / 2;
fprintf('\nOptimal alpha: %.4f\n', alpha_opt);
fprintf('f(alpha*) = %.4f\n\n', f_line(alpha_opt));

% Visualize
figure('Name', 'Example 3: Bisection Method', 'NumberTitle', 'off');
alpha_vals = linspace(0, 5, 200);
f_vals = arrayfun(f_line, alpha_vals);
plot(alpha_vals, f_vals, 'b-', 'LineWidth', 2);
hold on;
plot(alpha_opt, f_line(alpha_opt), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
title('Line Search Using Bisection Method', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('\alpha'); ylabel('f(\alpha)');
legend('f(\alpha)', 'Optimal \alpha*', 'Location', 'best');
grid on;

pause(2);

%% Example 4: Pattern Search Method

fprintf('=== Example 4: Pattern Search Method ===\n\n');

% Define the function
f_pattern = @(x) 9*x(1)^2 + x(2)^2 - 3*x(1)*x(2) + 15*x(1) - x(2) + 5;

% Gradient (for verification)
grad_pattern = @(x) [18*x(1) - 3*x(2) + 15; 
                     2*x(2) - 3*x(1) - 1];

% Initial point
x0 = [-150; 150];
epsilon = 0.01;

fprintf('Function: f(x) = 9x1^2 + x2^2 - 3x1*x2 + 15x1 - x2 + 5\n');
fprintf('Initial point: [%.2f, %.2f]\n', x0(1), x0(2));
fprintf('Tolerance: %.4f\n\n', epsilon);

% Pattern search implementation
x_current = x0;
k = 0;
max_iter = 50;
x_history = x0;

fprintf('Starting Pattern Search...\n');

while k < max_iter
    % Exploratory search along coordinate directions
    x_temp = x_current;
    
    % Direction 1 (e1 = [1; 0])
    f_along_e1 = @(alpha) f_pattern([x_current(1) + alpha; x_current(2)]);
    
    % Find best alpha1 using fminbnd
    alpha1 = fminbnd(f_along_e1, -200, 200);
    x_temp(1) = x_current(1) + alpha1;
    
    % Direction 2 (e2 = [0; 1])
    f_along_e2 = @(alpha) f_pattern([x_temp(1); x_current(2) + alpha]);
    
    % Find best alpha2 using fminbnd
    alpha2 = fminbnd(f_along_e2, -200, 200);
    x_temp(2) = x_current(2) + alpha2;
    
    % Pattern move direction
    d = x_temp - x_current;
    
    % Pattern move
    f_along_d = @(lambda) f_pattern(x_current + lambda * d);
    lambda_opt = fminbnd(f_along_d, -100, 100);
    
    x_next = x_current + lambda_opt * d;
    
    % Store history
    x_history = [x_history, x_next];
    
    % Check stopping criterion
    if norm(x_next - x_current) < epsilon
        fprintf('Converged at iteration %d\n', k+1);
        break;
    end
    
    x_current = x_next;
    k = k + 1;
    
    if mod(k, 10) == 0
        fprintf('Iteration %d: x = [%.4f, %.4f], f(x) = %.4f\n', ...
            k, x_current(1), x_current(2), f_pattern(x_current));
    end
end

fprintf('\nFinal solution: x* = [%.4f, %.4f]\n', x_current(1), x_current(2));
fprintf('Function value: f(x*) = %.4f\n', f_pattern(x_current));
fprintf('Gradient norm: ||grad f(x*)|| = %.6f\n\n', norm(grad_pattern(x_current)));

% Visualize convergence
figure('Name', 'Example 4: Pattern Search Convergence', 'NumberTitle', 'off');

% Create contour plot
[X1, X2] = meshgrid(-160:5:10, -10:5:160);
F = 9*X1.^2 + X2.^2 - 3*X1.*X2 + 15*X1 - X2 + 5;

contour(X1, X2, F, 50, 'LineWidth', 1);
hold on;
colormap('jet');
colorbar;

% Plot trajectory
plot(x_history(1,:), x_history(2,:), 'r-o', 'LineWidth', 2, ...
     'MarkerSize', 6, 'MarkerFaceColor', 'r');
plot(x0(1), x0(2), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
plot(x_current(1), x_current(2), 'mo', 'MarkerSize', 15, 'MarkerFaceColor', 'm');

title('Pattern Search: Convergence Path', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2');
legend('Contours', 'Path', 'Start', 'End', 'Location', 'best');
grid on;

pause(2);

%% Example 5: Steepest Descent Method

fprintf('=== Example 5: Steepest Descent Method ===\n\n');

% Same function as Pattern Search for comparison
f_steep = @(x) 9*x(1)^2 + x(2)^2 - 3*x(1)*x(2) + 15*x(1) - x(2) + 5;
grad_steep = @(x) [18*x(1) - 3*x(2) + 15; 
                   2*x(2) - 3*x(1) - 1];

% Initial point
x0_steep = [-150; 150];
epsilon_steep = 0.01;

fprintf('Using Steepest Descent Method\n');
fprintf('Initial point: [%.2f, %.2f]\n', x0_steep(1), x0_steep(2));
fprintf('Tolerance: %.4f\n\n', epsilon_steep);

% Steepest descent implementation
x_current_steep = x0_steep;
k_steep = 0;
max_iter_steep = 100;
x_history_steep = x0_steep;

fprintf('Starting Steepest Descent...\n');

while k_steep < max_iter_steep
    % Compute gradient at current point
    grad = grad_steep(x_current_steep);
    
    % Check stopping criterion
    if norm(grad) < epsilon_steep
        fprintf('Converged at iteration %d (gradient norm < epsilon)\n', k_steep);
        break;
    end
    
    % Steepest descent direction
    d_steep = -grad;
    
    % Line search: find optimal alpha
    f_along_d_steep = @(alpha) f_steep(x_current_steep + alpha * d_steep);
    
    % Use fminbnd for line search
    alpha_opt_steep = fminbnd(f_along_d_steep, -100, 100);
    
    % Update
    x_next_steep = x_current_steep + alpha_opt_steep * d_steep;
    
    % Store history
    x_history_steep = [x_history_steep, x_next_steep];
    
    % Check convergence
    if norm(x_next_steep - x_current_steep) < epsilon_steep
        fprintf('Converged at iteration %d (change in x < epsilon)\n', k_steep+1);
        break;
    end
    
    x_current_steep = x_next_steep;
    k_steep = k_steep + 1;
    
    if mod(k_steep, 10) == 0
        fprintf('Iteration %d: x = [%.4f, %.4f], f(x) = %.4f, ||grad|| = %.4f\n', ...
            k_steep, x_current_steep(1), x_current_steep(2), ...
            f_steep(x_current_steep), norm(grad));
    end
end

fprintf('\nFinal solution: x* = [%.4f, %.4f]\n', ...
    x_current_steep(1), x_current_steep(2));
fprintf('Function value: f(x*) = %.4f\n', f_steep(x_current_steep));
fprintf('Gradient norm: ||grad f(x*)|| = %.6f\n', ...
    norm(grad_steep(x_current_steep)));
fprintf('Number of iterations: %d\n\n', k_steep);

% Visualize convergence
figure('Name', 'Example 5: Steepest Descent Convergence', 'NumberTitle', 'off');

% Create contour plot
contour(X1, X2, F, 50, 'LineWidth', 1);
hold on;
colormap('jet');
colorbar;

% Plot trajectory
plot(x_history_steep(1,:), x_history_steep(2,:), 'b-o', 'LineWidth', 2, ...
     'MarkerSize', 6, 'MarkerFaceColor', 'b');
plot(x0_steep(1), x0_steep(2), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
plot(x_current_steep(1), x_current_steep(2), 'ro', 'MarkerSize', 15, ...
     'MarkerFaceColor', 'r');

title('Steepest Descent: Convergence Path', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2');
legend('Contours', 'Path', 'Start', 'End', 'Location', 'best');
grid on;

pause(2);

%% Example 6: Comparison of Methods

fprintf('=== Example 6: Comparing Pattern Search vs Steepest Descent ===\n\n');

figure('Name', 'Example 6: Method Comparison', 'NumberTitle', 'off');

% Create contour plot
contour(X1, X2, F, 50, 'LineWidth', 1);
hold on;
colormap('jet');
colorbar;

% Plot both trajectories
plot(x_history(1,:), x_history(2,:), 'r-o', 'LineWidth', 2, ...
     'MarkerSize', 5, 'MarkerFaceColor', 'r', 'DisplayName', 'Pattern Search');
plot(x_history_steep(1,:), x_history_steep(2,:), 'b-s', 'LineWidth', 2, ...
     'MarkerSize', 5, 'MarkerFaceColor', 'b', 'DisplayName', 'Steepest Descent');
plot(x0(1), x0(2), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g', ...
     'DisplayName', 'Start');
plot(-1, -1, 'mo', 'MarkerSize', 15, 'MarkerFaceColor', 'm', ...
     'DisplayName', 'True Optimum');

title('Comparison: Pattern Search vs Steepest Descent', ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2');
legend('Location', 'best');
grid on;

fprintf('Pattern Search:\n');
fprintf('  Iterations: %d\n', size(x_history, 2) - 1);
fprintf('  Final point: [%.4f, %.4f]\n', x_history(1,end), x_history(2,end));
fprintf('  Final f(x): %.6f\n\n', f_pattern(x_history(:,end)));

fprintf('Steepest Descent:\n');
fprintf('  Iterations: %d\n', size(x_history_steep, 2) - 1);
fprintf('  Final point: [%.4f, %.4f]\n', ...
    x_history_steep(1,end), x_history_steep(2,end));
fprintf('  Final f(x): %.6f\n\n', f_steep(x_history_steep(:,end)));

fprintf('True optimum: x* = [-1, -1], f(x*) = 0\n\n');

pause(2);

%% Example 7: 3D Visualization of the Function

fprintf('=== Example 7: 3D Function Visualization ===\n\n');

figure('Name', 'Example 7: 3D Function Surface', 'NumberTitle', 'off');

% Create finer mesh for better visualization
[X1_fine, X2_fine] = meshgrid(-5:0.2:5, -5:0.2:5);
F_fine = 9*X1_fine.^2 + X2_fine.^2 - 3*X1_fine.*X2_fine + ...
         15*X1_fine - X2_fine + 5;

% Surface plot
surf(X1_fine, X2_fine, F_fine, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
hold on;

% Plot the optimal point
plot3(-1, -1, 0, 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r');

colormap('jet');
colorbar;
title('f(x_1, x_2) = 9x_1^2 + x_2^2 - 3x_1x_2 + 15x_1 - x_2 + 5', ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x_1, x_2)');
view(45, 30);
lighting gouraud;
camlight;
grid on;

pause(2);

%% Example 8: Descent Direction Verification

fprintf('=== Example 8: Verifying Descent Direction ===\n\n');

% Test point
x_test = [1; 1];

% Compute gradient
grad_test = grad_steep(x_test);

fprintf('At point x = [%.2f, %.2f]\n', x_test(1), x_test(2));
fprintf('Gradient: [%.4f, %.4f]\n', grad_test(1), grad_test(2));

% Descent direction
d_descent = -grad_test;

fprintf('Descent direction d = -grad = [%.4f, %.4f]\n', ...
    d_descent(1), d_descent(2));

% Verify descent condition: grad' * d < 0
inner_prod = grad_test' * d_descent;

fprintf('Inner product grad^T * d = %.4f\n', inner_prod);

if inner_prod < 0
    fprintf('✓ This IS a descent direction (grad^T * d < 0)\n\n');
else
    fprintf('✗ This is NOT a descent direction\n\n');
end

% Visualize
figure('Name', 'Example 8: Descent Direction', 'NumberTitle', 'off');

% Create contour around test point
[X1_local, X2_local] = meshgrid(-1:0.1:3, -1:0.1:3);
F_local = 9*X1_local.^2 + X2_local.^2 - 3*X1_local.*X2_local + ...
          15*X1_local - X2_local + 5;

contour(X1_local, X2_local, F_local, 30, 'LineWidth', 1);
hold on;
colormap('jet');
colorbar;

% Plot gradient and descent direction
quiver(x_test(1), x_test(2), grad_test(1), grad_test(2), 0.3, ...
       'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Gradient');
quiver(x_test(1), x_test(2), d_descent(1), d_descent(2), 0.3, ...
       'b', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Descent Dir');

plot(x_test(1), x_test(2), 'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'k');

title('Gradient vs Descent Direction', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1'); ylabel('x_2');
legend('Location', 'best');
grid on;
axis equal;

pause(2);

%% Example 9: Convergence Analysis

fprintf('=== Example 9: Convergence Analysis ===\n\n');

% Analyze convergence of steepest descent
figure('Name', 'Example 9: Convergence Analysis', 'NumberTitle', 'off');

% Function values over iterations
f_values_steep = zeros(1, size(x_history_steep, 2));
for i = 1:size(x_history_steep, 2)
    f_values_steep(i) = f_steep(x_history_steep(:, i));
end

% Gradient norms over iterations
grad_norms_steep = zeros(1, size(x_history_steep, 2));
for i = 1:size(x_history_steep, 2)
    grad_norms_steep(i) = norm(grad_steep(x_history_steep(:, i)));
end

% Distance to optimum
dist_to_opt = zeros(1, size(x_history_steep, 2));
x_opt = [-1; -1];
for i = 1:size(x_history_steep, 2)
    dist_to_opt(i) = norm(x_history_steep(:, i) - x_opt);
end

% Plot
subplot(3,1,1);
semilogy(0:length(f_values_steep)-1, f_values_steep, 'b-o', ...
         'LineWidth', 2, 'MarkerSize', 4);
title('Function Value vs Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('f(x^{(k)})');
grid on;

subplot(3,1,2);
semilogy(0:length(grad_norms_steep)-1, grad_norms_steep, 'r-s', ...
         'LineWidth', 2, 'MarkerSize', 4);
title('Gradient Norm vs Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('||\nabla f(x^{(k)})||');
grid on;

subplot(3,1,3);
semilogy(0:length(dist_to_opt)-1, dist_to_opt, 'g-d', ...
         'LineWidth', 2, 'MarkerSize', 4);
title('Distance to Optimum vs Iteration', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Iteration k');
ylabel('||x^{(k)} - x^*||');
grid on;

fprintf('Convergence Analysis:\n');
fprintf('  Initial f(x): %.4f\n', f_values_steep(1));
fprintf('  Final f(x): %.6f\n', f_values_steep(end));
fprintf('  Optimal f(x*): 0\n');
fprintf('  Reduction: %.4f -> %.6f\n\n', ...
    f_values_steep(1), f_values_steep(end));

pause(2);

%% Example 10: Eigenvalue Analysis for Convexity

fprintf('=== Example 10: Eigenvalue Analysis ===\n\n');

% For the homework function
syms x1_sym x2_sym;
f_sym = 9*x1_sym^2 + x2_sym^2 - 3*x1_sym*x2_sym + 15*x1_sym - x2_sym + 5;

% Compute Hessian symbolically
grad_sym = [diff(f_sym, x1_sym); diff(f_sym, x2_sym)];
hess_sym = [diff(grad_sym(1), x1_sym), diff(grad_sym(1), x2_sym);
            diff(grad_sym(2), x1_sym), diff(grad_sym(2), x2_sym)];

fprintf('Hessian of f(x1, x2):\n');
disp(hess_sym);

% Convert to numeric
H_numeric = double(hess_sym);

fprintf('Numeric Hessian:\n');
disp(H_numeric);

% Compute eigenvalues
eigenvals = eig(H_numeric);
fprintf('Eigenvalues: %.4f, %.4f\n', eigenvals(1), eigenvals(2));

% Compute LPMs
lpm1 = H_numeric(1,1);
lpm2 = det(H_numeric);

fprintf('1st LPM = %.2f\n', lpm1);
fprintf('2nd LPM = %.2f\n', lpm2);

if all(eigenvals > 0) && lpm1 > 0 && lpm2 > 0
    fprintf('\n=> Function is STRICTLY CONVEX (everywhere)\n');
    fprintf('=> The stationary point is a GLOBAL MINIMUM\n\n');
elseif all(eigenvals >= 0)
    fprintf('\n=> Function is CONVEX\n\n');
else
    fprintf('\n=> Function is NOT convex\n\n');
end

pause(2);

%% Summary and Homework Solution

fprintf('========================================\n');
fprintf('HOMEWORK #3 SOLUTION SUMMARY\n');
fprintf('========================================\n\n');

fprintf('Problem: Implement Steepest Descent on:\n');
fprintf('f(x1, x2) = 9x1^2 + x2^2 - 3x1*x2 + 15x1 - x2 + 5\n');
fprintf('Starting point: x0 = [-150, 150]\n');
fprintf('Tolerance: epsilon = 0.01\n\n');

fprintf('SOLUTION:\n');
fprintf('Method: Steepest Descent with Exact Line Search (fminbnd)\n');
fprintf('Final point: x* = [%.4f, %.4f]\n', ...
    x_current_steep(1), x_current_steep(2));
fprintf('Function value: f(x*) = %.6f\n', f_steep(x_current_steep));
fprintf('Gradient norm: ||grad f(x*)|| = %.8f\n', ...
    norm(grad_steep(x_current_steep)));
fprintf('Number of iterations: %d\n\n', k_steep);

fprintf('Verification:\n');
fprintf('Analytical optimum: x* = [-1, -1]\n');
fprintf('Analytical f(x*) = 0\n');
fprintf('Error in x: ||x_computed - x_analytical|| = %.6f\n', ...
    norm(x_current_steep - [-1; -1]));
fprintf('Error in f: |f_computed - f_analytical| = %.6f\n\n', ...
    abs(f_steep(x_current_steep) - 0));

fprintf('Convexity Analysis:\n');
fprintf('Hessian is positive definite (all eigenvalues > 0)\n');
fprintf('=> Function is strictly convex\n');
fprintf('=> Stationary point is the global minimum\n\n');

fprintf('========================================\n');
fprintf('All examples completed successfully!\n');
fprintf('========================================\n\n');

fprintf('Key Takeaways:\n');
fprintf('1. Steepest Descent: Use negative gradient as direction\n');
fprintf('2. Pattern Search: Coordinate-wise search + pattern move\n');
fprintf('3. Line Search: Find optimal steplength (exact or approximate)\n');
fprintf('4. Convexity: Check Hessian definiteness via LPMs or eigenvalues\n');
fprintf('5. Convergence: Monitor function values, gradient norm, and distance\n\n');

fprintf('Thank you for using these implementations!\n');
fprintf('IE 317 - Nonlinear Optimization - Lecture 4\n');
