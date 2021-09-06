## Basic understanding of logit regression. 

using DataFrames
using CSV
using GLM
using FreqTables

obs = CSV.File("./logit_regression/logit_regression_data.csv", header=true) |> DataFrame

## summary of methods.
## want to model variable p in range (0, 1). This is really hard to do. 
## probability is linked with odds ratio = prob of success / prob of failure
## taking the log of this is the log odds ratio. 
## how do we transform p ∈ (0, 1) to odds ∈ (0, ∞)
## then take the log of odds to map to (-∞, ∞)
## This transformation is called logit transformation. The other common choice is the probit transformation,

## so regression of p is 
## log(p/(1-p)) =  linear predictors

## now lets see what this means in the sample data.  
# for the purpose of illustration.
# The data set has 200 observations and the outcome variable used will be hon, indicating if a student is in
# an honors class or not. So our p = prob(hon=1).  

# Model 1, no predictors
ols = glm(@formula(hon ~ 1), obs, Binomial(), LogitLink())
coef(ols) # so the intercept here is the log odds -1.12
# taking the exponential gives us the odds ratio. I.e. what are the odds of getting into a honors class
exp.(coef(ols))
exp.(coef(ols)) ./ (1 .+ exp.(coef(ols))) # we can get p back as well

# we can verify all of this by looking at a frequnecy table
# lets look at the FreqTables
#freqtable(obs, :hon)

# Model 2, a single predictor: female
ols = glm(@formula(hon ~ female), obs, Binomial(), LogitLink())
coef(ols)   ## gets the log odds
exp.(coef(ols)) ## gets the odds ratio 
## and can even calculate the prob given whether person is male or female

# here theintercept of -1.471 is the log odds for males since male is the reference group (female = 0). 
# so this means the odds ratio is 0.23 

# Using the frequency for males, we can confirm this: log(.23) = -1.47.

# Model 3, using math
ols = glm(@formula(hon ~ math), obs, Binomial(), LogitLink())
coef(ols)   ## gets the log odds
exp.(coef(ols)) ## gets the odds ratio 
## and can even calculate the prob given math score. 

# Model 3, using combination
ols = glm(@formula(hon ~ math + female + read), obs, Binomial(), LogitLink())
coef(ols)   ## gets the log odds
exp.(coef(ols)) ## gets the odds ratio 
## and can even calculate the prob given a different combination of predictors

# This fitted model says that, holding math and reading at a fixed value, the odds of getting into an
# honors class for females (female = 1)over the odds of getting into an honors class for males (female = 0)
# is exp(.979948) = 2.66

# Model 4, interaction term
ols = glm(@formula(hon ~ math + female + female*math), obs, Binomial(), LogitLink())
coef(ols)   ## gets the log odds
exp.(coef(ols)) ## gets the odds ratio 
