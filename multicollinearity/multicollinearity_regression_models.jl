# Multicollinearity Test for regression models:
# see associated pdf and data file
# overview of notes: 
# if doing regression, the coefficient of each predictor (indepedent) variable tells you how much on average the response changes
# when the variable changes by a unit. however, when variables are corellated (they are not really independent) it makes inference harder

using RCall
using DataFrames
using CSV
using GLM
using Statistics

# read the first 4 columns (not the columns with S)
df = CSV.read("multicollinearity_regression_models_data.csv" , header=1, select=(i, name) -> i < 5 && return true) |> DataFrame#, select=(i, name) -> i < 5 && return true)
insertcols!(df, :pfat_weight => df.pfat .* df.weight) # create interaction term manually

rmodel = lm(@formula(neck ~ pfat + weight + activity + pfat_weight), df) 
# lets calculate the Variance Inflation numbers
# using r2 of the model, calculate VIF

rmodel1 = lm(@formula(pfat ~  weight + activity + pfat_weight), df)
rmodel2 = lm(@formula(weight ~ pfat + activity + pfat_weight), df)
rmodel3 = lm(@formula(activity ~ pfat + weight + pfat_weight), df)
rmodel4 = lm(@formula(pfat_weight ~ pfat + weight + activity), df)
rmodels = (rmodel1, rmodel2, rmodel3, rmodel4)
vifs = @. 1 / (1 - r2(rmodels))

# node that for activity, VIF is 1.05 so this variable has no mulitcollinearity. 
# for the other ones, there seems to be definitely multicollinearity
# first, let's remove the structural multicollinearity from the interaction term. 
# to do tht, we will first mean center the data. 

# there is a bug in Query that messes up the dataframe types and GLM can't read it anymore
# https://discourse.julialang.org/t/query-jl-mutate-command-does-not-preserve-column-types/41231/4
# aa = mean(df.pfat)::Float64
# ab = mean(df.weight)::Float64
# ac = mean(df.activity)::Float64
# ad = mean(df.pfat_weight)::Float64
# df_s = df |> @mutate(pfat_s .= _.pfat .- aa, 
#               weight_s .= _.weight .- ab, 
#               activity_s .= _.activity .- ac, 
#               pfat_weight_s .= _.pfat_weight .- ad) |> DataFrame
             
# equivalent linq
# @from i in df begin
# @select {i..., pfat_s = mean(df.pfat)}
# @collect DataFrame
# end

# lets use standard method to add columns
df[!, :neck_s] .= mean(df.neck) .- df.neck 
df[!, :pfat_s] .= mean(df.pfat) .- df.pfat 
df[!, :weight_s] .= mean(df.weight) .- df.weight 
df[!, :activity_s] .= mean(df.activity) .- df.activity 
df[!, :pfat_weight_s] .=  df.pfat_s .* df.weight_s

smodel = lm(@formula(neck ~ pfat_s + weight_s + activity_s + pfat_weight_s), df) 
# lets calculate the Variance Inflation numbers
# using r2 of the model, calculate VIF
smodel1 = lm(@formula(pfat_s ~  weight_s + activity_s + pfat_weight_s), df)
smodel2 = lm(@formula(weight_s ~ pfat_s + activity_s + pfat_weight_s), df)
smodel3 = lm(@formula(activity_s ~ pfat_s + weight_s + pfat_weight_s), df)
smodel4 = lm(@formula(pfat_weight_s ~ pfat_s + weight_s + activity_s), df)
smodels = (smodel1, smodel2, smodel3, smodel4)
vifs = @. 1 / (1 - r2(smodels))
# (3.323870441091722, 4.745648167219831, 1.0530047498114123, 1.9910631706191224) 

# you see now how small the VIFs have gotten just by removing the multicollinearity
