# Bachelor's Thesis
## Mouse sleep analysis with machine learning

### Publishable summary
This project revolves around the analysis of EEG (electroencephalogram) and EMG (elec-
tromyogram) data from 250 mice, collected by UNIL researchers for a biological study on
sleep. The data involves three sleep stages: REM (rapid eye movement), NREM (non-rapid
eye movement), and Wake. However, the data is unbalanced, with REM representing only
about 5% of the total data. Due to the laborious nature of manual sleep data annotation,
machine learning techniques are explored to achieve more efficient data interpretation.
The data is transformed and subjected to feature engineering to extract relevant informa-
tion. Three initial experiments using a random forest classifier achieved excellent results in
classifying sleep stages for individual mice, entire breeds, and all mice from different breeds.
To address the issue of unbalanced labels, ensemble methods like the balanced random forest
and easy ensemble classifiers were tested, with the balanced random forest showing the most
promising results. A combined approach was adopted, where the first model classified wake
versus sleep, and the second model classified REM versus NREM. This combined approach
yielded performance similar to the balanced random forest. Feature importance ranking and
exploration were utilized to reduce the number of features and simplify the input, enhancing
the model’s efficiency.
Additionally, a comparison was made between the random forest classifier and the SVM
annotation method developed in the UNIL survey, which employed one SVM model for
each mouse. Cohen’s Kappa coefficient was used for this comparison, and both methods,
particularly SVM, showed high accuracy scores across all days. While the balanced random
forest exhibited slightly lower coefficients initially, its performance improved on subsequent
days, indicating potential for similar classification patterns and shared misclassifications
between the models.

## Project Organization
This project is organized into distinct parts, each represented by a corresponding folder in the repository https://github.com/magskwa/TB_sleep_analysis_with_ML. The repository includes the following folders:

1. Preparation: Contains the code to create the data frame used in this project.
2. Library: Contains Python files used for managing mice, breeds, data splitting, and feature computation.
3. Exploration: Includes the exploratory data analysis.
4. ClassificationScale: Covers the classification experiments for one mouse, one breed, and all mice.
5. EnsembleModels: Involves the exploration of models with better performance using ensemble methods.
6. CombinedModels: Contains the experiments with two models handling different tasks working together.
7. InputRelevance: Focuses on dimensionality reduction techniques.
8. Kappa: Involves the comparison of the best model found in this project, SVM, and the ground truth.




