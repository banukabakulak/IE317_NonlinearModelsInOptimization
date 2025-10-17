% =========================================================================
% IE 317 - Nonlinear Optimization
% Lecture 3: Convexity
% MATLAB Examples and Visualizations
% =========================================================================

clear all; close all; clc;

%% Example 1: Plotting Convex vs Non-Convex Functions
% This example compares a convex function (x^2) with a non-convex function (x^3)

fprintf('Running Example 1: Convex vs Non-Convex Functions...\n');

x = linspace(-2, 2, 100);

% Define functions
f = x.^2;  % Convex function
g = x.^3;  % Not convex (has inflection point)

% Plot
figure('Name', 'Example 1: Convex vs Non-Convex', 'NumberTitle', 'off');
subplot(1,2,1);
plot(x, f, 'b-', 'LineWidth', 2);
title('f(x) = x^2 (Convex)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('f(x)', 'FontSize', 11);
grid on;

subplot(1,2,2);
plot(x, g, 'r-', 'LineWidth', 2);
title('g(x) = x^3 (Not Convex)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('g(x)', 'FontSize', 11);
grid on;

fprintf('Example 1 completed!\n\n');
pause(2);

%% Example 2: Visualizing Convex Combinations
% Show that a convex combination of two points forms a line segment

fprintf('Running Example 2: Convex Combinations...\n');

% Two points in 2D
x1 = [1; 2]; 
x2 = [4; 5];

% Generate convex combinations for lambda in [0,1]
lambda = linspace(0, 1, 50);
points = zeros(2, length(lambda));

for i = 1:length(lambda)
    points(:,i) = lambda(i)*x1 + (1-lambda(i))*x2;
end

% Plot
figure('Name', 'Example 2: Convex Combinations', 'NumberTitle', 'off');
plot(points(1,:), points(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x1(1), x1(2), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'x_1');
plot(x2(1), x2(2), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'x_2');
text(x1(1)-0.3, x1(2), 'x_1', 'FontSize', 12, 'FontWeight', 'bold');
text(x2(1)+0.2, x2(2), 'x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Convex Combinations: \lambda x_1 + (1-\lambda) x_2, \lambda \in [0,1]', ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1', 'FontSize', 11); 
ylabel('x_2', 'FontSize', 11); 
grid on;
axis equal;

fprintf('Example 2 completed!\n\n');
pause(2);

%% Example 3: Checking Convexity Using Second Derivative
% Use symbolic math to compute derivatives and check convexity

fprintf('Running Example 3: Second Derivative Test...\n');

% Define symbolic function
syms x;
f = x^4 - 3*x^2 + 2*x + 1;

% Compute first and second derivatives
f_prime = diff(f, x);
f_double_prime = diff(f_prime, x);

% Display
fprintf('Function: f(x) = '); disp(f);
fprintf('First derivative: f''(x) = '); disp(f_prime);
fprintf('Second derivative: f''''(x) = '); disp(f_double_prime);

% Check convexity at specific points
x_vals = -2:0.5:2;
second_deriv_vals = double(subs(f_double_prime, x, x_vals));

fprintf('\nSecond derivative values at test points:\n');
for i = 1:length(x_vals)
    fprintf('x = %.1f: f''''(x) = %.2f\n', x_vals(i), second_deriv_vals(i));
end

if all(second_deriv_vals >= 0)
    fprintf('\n✓ Function is CONVEX on the interval [-2, 2]\n');
else
    fprintf('\n✗ Function is NOT convex on the interval [-2, 2]\n');
    fprintf('  (Some second derivative values are negative)\n');
end

% Plot the function and its second derivative
x_plot = linspace(-2, 2, 200);
f_vals = double(subs(f, x, x_plot));
f_pp_vals = double(subs(f_double_prime, x, x_plot));

figure('Name', 'Example 3: Second Derivative Test', 'NumberTitle', 'off');
subplot(2,1,1);
plot(x_plot, f_vals, 'b-', 'LineWidth', 2);
title('f(x) = x^4 - 3x^2 + 2x + 1', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('f(x)', 'FontSize', 11);
grid on;

subplot(2,1,2);
plot(x_plot, f_pp_vals, 'r-', 'LineWidth', 2);
hold on;
plot(x_plot, zeros(size(x_plot)), 'k--', 'LineWidth', 1);
title('f''''(x) = 12x^2 - 6', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('f''''(x)', 'FontSize', 11);
grid on;
legend('f''''(x)', 'y = 0', 'Location', 'best');

fprintf('Example 3 completed!\n\n');
pause(2);

%% Example 4: Sum of Convex Functions (Example 11 from notes)
% Demonstrate that the sum of convex functions is convex

fprintf('Running Example 4: Sum of Convex Functions...\n');

% Define x range
x = linspace(0, 6, 200);

% Two convex functions
f = (x - 2).^2 + 3;
g = (x - 4).^2 + 2;

% Sum
h = f + g;

% Plot
figure('Name', 'Example 4: Sum of Convex Functions', 'NumberTitle', 'off');
plot(x, f, 'b-', 'LineWidth', 2, 'DisplayName', 'f(x) = (x-2)^2 + 3');
hold on;
plot(x, g, 'r-', 'LineWidth', 2, 'DisplayName', 'g(x) = (x-4)^2 + 2');
plot(x, h, 'g-', 'LineWidth', 2.5, 'DisplayName', 'h(x) = f(x) + g(x)');
title('Sum of Convex Functions is Convex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('y', 'FontSize', 11); 
legend('show', 'Location', 'best', 'FontSize', 10);
grid on;

fprintf('Example 4 completed!\n');
fprintf('✓ The sum h(x) = f(x) + g(x) is also convex!\n\n');
pause(2);

%% Example 5: Difference of Convex Functions (Example 12 from notes)
% Show that the difference of convex functions may NOT be convex

fprintf('Running Example 5: Difference of Convex Functions...\n');

% Define x range
x = linspace(0, 5, 200);

% Two convex functions
f = (x - 2).^2 + 3;
g = (2*x - 4).^2 + 2;

% Difference
h = f - g;

% Compute second derivative of h numerically
dx = x(2) - x(1);
h_prime = gradient(h, dx);
h_double_prime = gradient(h_prime, dx);

% Plot
figure('Name', 'Example 5: Difference of Convex Functions', 'NumberTitle', 'off');

subplot(3,1,1);
plot(x, f, 'b-', 'LineWidth', 2);
hold on;
plot(x, g, 'r-', 'LineWidth', 2);
legend('f(x) = (x-2)^2 + 3 (convex)', 'g(x) = (2x-4)^2 + 2 (convex)', ...
       'Location', 'best');
title('Both Functions are Convex', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 11);
grid on;

subplot(3,1,2);
plot(x, h, 'k-', 'LineWidth', 2);
title('h(x) = f(x) - g(x) - NOT Convex!', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('h(x)', 'FontSize', 11); 
grid on;

subplot(3,1,3);
plot(x, h_double_prime, 'm-', 'LineWidth', 2);
hold on;
plot(x, zeros(size(x)), 'k--', 'LineWidth', 1);
title('h''''(x) - Notice negative values!', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('h''''(x)', 'FontSize', 11);
grid on;

fprintf('Example 5 completed!\n');
fprintf('✗ The difference h(x) = f(x) - g(x) is NOT convex!\n');
fprintf('  (h''''(x) has negative values)\n\n');
pause(2);

%% Example 6: Maximum of Convex Functions (Example 13 from notes)
% Demonstrate that the maximum of convex functions is convex

fprintf('Running Example 6: Maximum of Convex Functions...\n');

% Define x range
x = linspace(0, 6, 200);

% Two convex functions
f = (x - 2).^2 + 3;
g = (x - 4).^2 + 2;

% Maximum
h = max(f, g);

% Plot
figure('Name', 'Example 6: Maximum of Convex Functions', 'NumberTitle', 'off');
plot(x, f, 'b--', 'LineWidth', 1.5, 'DisplayName', 'f(x) = (x-2)^2 + 3');
hold on;
plot(x, g, 'r--', 'LineWidth', 1.5, 'DisplayName', 'g(x) = (x-4)^2 + 2');
plot(x, h, 'k-', 'LineWidth', 2.5, 'DisplayName', 'h(x) = max{f(x), g(x)}');
title('Maximum of Convex Functions is Convex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('y', 'FontSize', 11); 
legend('show', 'Location', 'best', 'FontSize', 10);
grid on;

fprintf('Example 6 completed!\n');
fprintf('✓ The maximum h(x) = max{f(x), g(x)} is convex!\n\n');
pause(2);

%% Example 7: 3D Visualization of Convex Function
% Plot a 2D convex function in 3D

fprintf('Running Example 7: 3D Convex Function Visualization...\n');

% Create grid
[X, Y] = meshgrid(-3:0.1:3, -3:0.1:3);

% Define convex function: f(x,y) = x^2 + y^2
Z = X.^2 + Y.^2;

% Plot surface
figure('Name', 'Example 7: 3D Convex Function', 'NumberTitle', 'off');
surf(X, Y, Z, 'EdgeColor', 'none');
colormap('jet');
title('Convex Function: f(x,y) = x^2 + y^2', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('y', 'FontSize', 11); 
zlabel('f(x,y)', 'FontSize', 11);
colorbar;
view(45, 30);
lighting gouraud;
camlight;

fprintf('Example 7 completed!\n\n');
pause(2);

%% Example 8: 3D Visualization with Level Sets
% Show convex function with level curves

fprintf('Running Example 8: 3D Function with Level Sets...\n');

% Create grid
[X, Y] = meshgrid(-3:0.15:3, -3:0.15:3);

% Define another convex function: f(x,y) = x^2 + 2*y^2
Z = X.^2 + 2*Y.^2;

% Plot
figure('Name', 'Example 8: Convex Function with Level Sets', 'NumberTitle', 'off');

% 3D surface
subplot(1,2,1);
surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
colormap('parula');
title('f(x,y) = x^2 + 2y^2', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('y', 'FontSize', 11); 
zlabel('f(x,y)', 'FontSize', 11);
view(45, 30);
lighting gouraud;
camlight;

% Contour plot (level sets)
subplot(1,2,2);
contour(X, Y, Z, 20, 'LineWidth', 1.5);
colormap('parula');
colorbar;
title('Level Sets (All Convex!)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11); 
ylabel('y', 'FontSize', 11);
axis equal;
grid on;

fprintf('Example 8 completed!\n');
fprintf('✓ All level sets of a convex function are convex!\n\n');
pause(2);

%% Example 9: LP Feasible Region Visualization
% Plot LP feasible region (convex polytope)

fprintf('Running Example 9: LP Feasible Region...\n');

% Define constraints:
% x1 + x2 <= 8
% 2*x1 + x2 <= 12
% x1, x2 >= 0

x = 0:0.1:10;

% Constraint boundaries
y1 = 8 - x;        % from x1 + x2 = 8
y2 = 12 - 2*x;     % from 2*x1 + x2 = 12

% Find vertices of feasible region
vertices = [0, 0;      % origin
            0, 8;      % intersection with x2 axis
            4, 4;      % intersection of two constraints
            6, 0];     % intersection with x1 axis

figure('Name', 'Example 9: LP Feasible Region', 'NumberTitle', 'off');

% Fill feasible region
fill(vertices(:,1), vertices(:,2), 'b', 'FaceAlpha', 0.3, ...
     'DisplayName', 'Feasible Region');
hold on;

% Plot constraint lines
plot(x, y1, 'r-', 'LineWidth', 2, 'DisplayName', 'x_1 + x_2 \leq 8');
plot(x, y2, 'g-', 'LineWidth', 2, 'DisplayName', '2x_1 + x_2 \leq 12');

% Plot vertices
plot(vertices(:,1), vertices(:,2), 'ko', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'k', 'DisplayName', 'Extreme Points');

% Labels
for i = 1:size(vertices, 1)
    text(vertices(i,1)+0.3, vertices(i,2)+0.3, ...
         sprintf('e_%d', i), 'FontSize', 11, 'FontWeight', 'bold');
end

xlim([0 10]); 
ylim([0 10]);
title('LP Feasible Region (Convex Set)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x_1', 'FontSize', 11); 
ylabel('x_2', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 10);
grid on;
axis equal;

fprintf('Example 9 completed!\n');
fprintf('✓ LP feasible region is the convex hull of extreme points!\n\n');
pause(2);

%% Example 10: Comparing Convex, Concave, and Non-Convex
% Visual comparison of different function types

fprintf('Running Example 10: Convex vs Concave vs Non-Convex...\n');

x = linspace(-2, 2, 200);

% Three different functions
f_convex = x.^2;                          % Convex
f_concave = -x.^2 + 4;                    % Concave
f_neither = x.^3 - 3*x;                   % Neither

figure('Name', 'Example 10: Function Types', 'NumberTitle', 'off');

% Convex
subplot(3,1,1);
plot(x, f_convex, 'b-', 'LineWidth', 2);
hold on;
% Show chord above function
x1 = -1; x2 = 1.5;
y1 = x1^2; y2 = x2^2;
plot([x1 x2], [y1 y2], 'r--', 'LineWidth', 2);
plot([x1 x2], [y1 y2], 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('CONVEX: f(x) = x^2 (chord above function)', ...
      'FontSize', 11, 'FontWeight', 'bold');
ylabel('f(x)', 'FontSize', 10);
grid on;

% Concave
subplot(3,1,2);
plot(x, f_concave, 'g-', 'LineWidth', 2);
hold on;
% Show chord below function
x1 = -1; x2 = 1.5;
y1 = -x1^2 + 4; y2 = -x2^2 + 4;
plot([x1 x2], [y1 y2], 'r--', 'LineWidth', 2);
plot([x1 x2], [y1 y2], 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('CONCAVE: f(x) = -x^2 + 4 (chord below function)', ...
      'FontSize', 11, 'FontWeight', 'bold');
ylabel('f(x)', 'FontSize', 10);
grid on;

% Neither
subplot(3,1,3);
plot(x, f_neither, 'm-', 'LineWidth', 2);
hold on;
% Show chord that intersects function
x1 = -1.5; x2 = 1.5;
y1 = x1^3 - 3*x1; y2 = x2^3 - 3*x2;
plot([x1 x2], [y1 y2], 'r--', 'LineWidth', 2);
plot([x1 x2], [y1 y2], 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('NEITHER: f(x) = x^3 - 3x (chord crosses function)', ...
      'FontSize', 11, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 10);
ylabel('f(x)', 'FontSize', 10);
grid on;

fprintf('Example 10 completed!\n\n');
pause(2);

%% Example 11: Checking Convexity of NLP (Example 14 from notes)
% Check if the given NLP is a convex programming problem

fprintf('Running Example 11: Checking NLP Convexity...\n');

% NLP: min f(x) = x^2 + 1
%      s.t. x <= 3
%           2*x^4 <= 1
%           x in R

% Check objective function
syms x;
f_obj = x^2 + 1;
f_obj_pp = diff(diff(f_obj, x), x);

fprintf('\nObjective function: f(x) = x^2 + 1\n');
fprintf('Second derivative: f''''(x) = '); disp(f_obj_pp);

if double(f_obj_pp) > 0
    fprintf('✓ Objective function is CONVEX (f''''(x) = 2 > 0)\n\n');
else
    fprintf('✗ Objective function is NOT convex\n\n');
end

% Check feasible region
fprintf('Checking feasible region:\n');
fprintf('Constraint 1: x <= 3 (half-space) ✓ CONVEX\n');
fprintf('Constraint 2: 2*x^4 <= 1 (x^4 <= 0.5)\n');

% Visualize S2 = {x : 2*x^4 <= 1}
x_vals = linspace(-2, 2, 1000);
constraint2 = 2*x_vals.^4;

figure('Name', 'Example 11: Feasible Region Analysis', 'NumberTitle', 'off');

subplot(2,1,1);
plot(x_vals, constraint2, 'b-', 'LineWidth', 2);
hold on;
plot(x_vals, ones(size(x_vals)), 'r--', 'LineWidth', 2);
title('Constraint: 2x^4 \leq 1', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11);
ylabel('2x^4', 'FontSize', 11);
legend('2x^4', 'y = 1', 'Location', 'best');
grid on;

% Show S2 on number line
subplot(2,1,2);
x_bound = (0.5)^(1/4);  % approximately 0.841
feasible_x = x_vals(abs(x_vals) <= x_bound);

plot([-3, 3], [0, 0], 'k-', 'LineWidth', 1);
hold on;
plot(feasible_x, zeros(size(feasible_x)), 'b-', 'LineWidth', 8);
plot([-x_bound, x_bound], [0, 0], 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
ylim([-0.5, 0.5]);
title('Feasible Set S_2 = {x : 2x^4 \leq 1}', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11);
text(0, -0.3, sprintf('S_2 = [%.3f, %.3f]', -x_bound, x_bound), ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

fprintf('\nS_2 = {x : |x| <= %.3f}\n', x_bound);
fprintf('✗ S_2 is NOT a convex set!\n');
fprintf('  (A convex combination of boundary points may not be in S_2)\n\n');

fprintf('CONCLUSION:\n');
fprintf('Objective: CONVEX ✓\n');
fprintf('Feasible region: NOT CONVEX ✗\n');
fprintf('=> The NLP is NOT a convex programming problem!\n\n');

pause(2);

%% Example 12: Demonstrating Jensen's Inequality
% Visual proof of convexity through Jensen's inequality

fprintf('Running Example 12: Jensen''s Inequality...\n');

% Convex function
f_func = @(x) 0.5*x.^2 - x + 2;

x = linspace(-1, 5, 200);
f_vals = f_func(x);

% Choose two points
x1 = 1;
x2 = 4;
f_x1 = f_func(x1);
f_x2 = f_func(x2);

% Convex combination with lambda = 0.3
lambda = 0.3;
x_conv = lambda*x1 + (1-lambda)*x2;
f_x_conv = f_func(x_conv);
f_chord = lambda*f_x1 + (1-lambda)*f_x2;

figure('Name', 'Example 12: Jensen''s Inequality', 'NumberTitle', 'off');

% Plot function
plot(x, f_vals, 'b-', 'LineWidth', 2.5);
hold on;

% Plot chord
plot([x1, x2], [f_x1, f_x2], 'r-', 'LineWidth', 2);

% Plot points
plot(x1, f_x1, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot(x2, f_x2, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot(x_conv, f_x_conv, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
plot(x_conv, f_chord, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw vertical line showing the gap
plot([x_conv, x_conv], [f_x_conv, f_chord], 'g-', 'LineWidth', 2);

% Labels
text(x1, f_x1-0.3, 'f(x_1)', 'FontSize', 11, 'FontWeight', 'bold');
text(x2, f_x2-0.3, 'f(x_2)', 'FontSize', 11, 'FontWeight', 'bold');
text(x_conv-0.5, f_chord+0.3, '\lambda f(x_1)+(1-\lambda)f(x_2)', ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
text(x_conv-0.5, f_x_conv-0.3, 'f(\lambda x_1+(1-\lambda)x_2)', ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', 'm');
text(x_conv+0.1, (f_x_conv+f_chord)/2, '\leftarrow Gap', ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', 'g');

title(sprintf('Jensen''s Inequality: f(\\lambda x_1+(1-\\lambda)x_2) \\leq \\lambda f(x_1)+(1-\\lambda)f(x_2), \\lambda=%.1f', lambda), ...
      'FontSize', 11, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 11);
ylabel('f(x)', 'FontSize', 11);
grid on;

fprintf('Example 12 completed!\n');
fprintf('✓ Gap shows f(λx₁+(1-λ)x₂) ≤ λf(x₁)+(1-λ)f(x₂)\n\n');

pause(2);

%% Example 13: Common Convex Functions Gallery
% Showcase multiple common convex functions

fprintf('Running Example 13: Common Convex Functions...\n');

x = linspace(0.1, 5, 200);

figure('Name', 'Example 13: Common Convex Functions', 'NumberTitle', 'off');

% Quadratic
subplot(2,3,1);
plot(x, x.^2, 'b-', 'LineWidth', 2);
title('f(x) = x^2', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

% Exponential
subplot(2,3,2);
plot(x, exp(x), 'r-', 'LineWidth', 2);
title('f(x) = e^x', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

% Negative log
subplot(2,3,3);
plot(x, -log(x), 'g-', 'LineWidth', 2);
title('f(x) = -log(x)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

% x*log(x)
subplot(2,3,4);
plot(x, x.*log(x), 'm-', 'LineWidth', 2);
title('f(x) = x log(x)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

% Absolute value
x_abs = linspace(-3, 3, 200);
subplot(2,3,5);
plot(x_abs, abs(x_abs), 'c-', 'LineWidth', 2);
title('f(x) = |x|', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

% Max function
subplot(2,3,6);
f1 = (x - 2).^2;
f2 = 0.5*(x - 3).^2 + 1;
plot(x, max(f1, f2), 'k-', 'LineWidth', 2);
title('f(x) = max{f_1(x), f_2(x)}', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('x'); ylabel('f(x)'); grid on;

fprintf('Example 13 completed!\n');
fprintf('✓ All displayed functions are convex!\n\n');

%% Summary
fprintf('========================================\n');
fprintf('All examples completed successfully!\n');
fprintf('========================================\n\n');

fprintf('Key Takeaways:\n');
fprintf('1. Convex functions: chord lies above the function\n');
fprintf('2. Check convexity: f''''(x) >= 0 (second derivative test)\n');
fprintf('3. Sum of convex functions is convex\n');
fprintf('4. Max of convex functions is convex\n');
fprintf('5. Difference may NOT be convex\n');
fprintf('6. LP feasible regions are always convex\n');
fprintf('7. Convex optimization: local optimum = global optimum!\n\n');

fprintf('Thank you for using these examples!\n');
fprintf('IE 317 - Nonlinear Optimization\n');
