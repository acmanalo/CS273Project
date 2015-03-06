%% Load data
kaggleX = load('data/kaggle.X1.train.txt');
kaggleY = load('data/kaggle.Y.train.txt');
kaggleTestData = load('data/kaggle.X1.test.txt');

iris = load('data/iris.txt');
X = iris(:, 1:end-1);
Y = iris(:, end);

%% Split data
[xtr, xte, ytr, yte] = splitData(kaggleX, kaggleY, .05);
[Xtr, Xte, Ytr, Yte] = splitData(X, Y, .75);

%% Run on IRIS
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData iris X Y
rand('state',0)

nPoints = size(Xte, 1);

predicitions = zeros(nPoints, 1);
for i = 1:nPoints
    predictions(i) = lwrPredict(Xtr, Ytr, Xte(i, :), 1);
end

fig()
hold on
plot(predictions);
plot(Yte);

%% Run
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData iris X Y
rand('state',0)

[xtr2, xte2, ytr2, yte2] = splitData(xtr, ytr, .66);

[xtr2, mu, sigma] = zscore(xtr2);
xte2 = normalize(xte2, mu, sigma);

nPoints = size(xte2, 1);

predictions = zeros(nPoints, 1);
for i = 1:nPoints
    predictions(i) = lwrPredict(xtr2, ytr2, xte2(i, :), 1);
end

%% Linear Regress from class
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData iris X Y
rand('state',0)

[xtr2, xte2, ytr2, yte2] = splitData(xtr(:, 3:end), ytr, .66);

[xtr2, mu, sigma] = zscore(xtr2);
xte2 = normalize(xte2, mu, sigma);

lr = linearRegress(xtr2, ytr2);
lr = train(lr, xtr2, ytr2, 0);

mse(lr, xte2, yte2)