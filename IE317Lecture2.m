%[text] ## 3D Plot $f(x) = (x-2)^2 + (y-3)^2$ function
%%
x = linspace(0,10,100);
y = linspace(0,10,100);

myFunc1 = power((x.^1 - 2), 2) + power((y.^1 - 3), 2);

plot3(x,y,myFunc1) 
%%
%[text] ## Plot the 3D surface of the function
%%
[X,Y] = meshgrid(-10:0.5:20,-10:0.5:20);

myFunc1surface = power((X - 2), 2) + power((Y - 3), 2);

surf(X, Y, myFunc1surface) 

%%
%[text] ## Plot with the constraint $5x + 10y \\leq 50$
%%
[X,Y] = meshgrid(-10:0.5:20,-10:0.5:20);

myFunc4 = power((X - 2), 2) + power((Y - 3), 2);

constr1 = 5 * X + 10 * Y - 50;

hold on 
surf(X, Y, myFunc4) 

mesh(X, Y, constr1) 

hold off

%%
%[text] ## A Feasible Region $x^2 + y^2 \\leq 1$without any Extreme Points 
%%
[X,Y] = meshgrid(-10:0.5:20,-10:0.5:20);

constr2 = power(X, 2) + power(Y, 2) - 1;

mesh(X, Y, constr2); 
%%
%[text] ## **2D Plot**$f(x) = (x^2 - 1)^3$ **function**
x = linspace(-10,10,100);

myFunc2 = power((x.^2 - 1), 3);

plot(myFunc2) 
%%
%[text] ## Plot the 3D surface of the function
%%
[X,Y] = meshgrid(-10:0.1:10,-2:0.5:10);

myFunc2surface = power((X.^2 - 1), 3);

surf(X, Y, myFunc2surface); 

xlabel('x-axis') 
ylabel('y-axis') 

