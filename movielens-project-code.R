#############################################################
# Create edx set, validation set, and submission file
#############################################################

#Dataset build--------------------------------------------------------

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#end dataset build-------------------------------------------------------------------------

#basic statistics for the training data set -----------------------------------------------
#size of training dataset
dim(edx)

#size of test data set
dim(validation)

#counts by rating type
table(edx$rating)

#number of ratings equal to 3
edx %>% filter(rating == 3) %>% tally()

#number of distinct movies
# training
n_distinct(edx$movieId)

# test
n_distinct(validation$movieId)

#number of distinct users
# training
n_distinct(edx$userId)

# test
n_distinct(validation$userId)

#view of matrix sith 100 randomally selected users and movies
# shows an example of the amount of missing ratings for users
users <- sample(unique(edx$userId), 100)

edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

#chart with the distribution of ratings per movie
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#chart with the distibution of ratings per user
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

#movie with greatest number of ratings, top 10
edx %>% 
  group_by(title) %>% 
  summarize(n=n()) %>% 
  arrange(desc(n)) %>% 
  slice(1:10) %>%
  knitr::kable()

#and fewest ratings, top 10
edx %>% 
  group_by(title) %>% 
  summarize(n=n()) %>% 
  arrange(n) %>%
  slice(1:10) %>%
  knitr::kable()

#the 5 most given ratings
edx %>% 
  group_by(rating) %>% 
  summarize(n=n()) %>%
  arrange(desc(n)) %>%
  slice(1:5)%>% knitr::kable()

#end of basic statistics ------------------------------------------------------------------

#------------------Model development and RMSE cacluations ---------------------------------

#residual mean squared error to see how good the model works
RMSE <- function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#mean rating for the entire training set---------------------------------------------------
mu_hat <- mean(edx$rating)
mu_hat

#RMSE on test set using mu_hat
naive_rmse <- RMSE(validation$rating,mu_hat)
naive_rmse

#create a table to store the results
rmse_results <- data_frame(method="Just the average", rmse=naive_rmse)
#-----------------------------------------------------------------------------------------
#run the model with any other single number and get a higher RMSE, for example, try 4
predictions <- rep(4,nrow(validation))
single_number_rmse <- RMSE(validation$rating,predictions)
single_number_rmse
#adding result to RMSE results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Using another single number as the predictor (4 for this example)",
                                     rmse=single_number_rmse))
#creating movie effect variable b_i-------------------------------------------------------
# Model: y_u_i=mu+b_i+e_u_i

#calculating the average for the whole training set
mu <- mean(edx$rating)

#looking at the mean rating by movie
edx %>% group_by(movieId) %>% 
  summarize(avg=mean(rating)) %>%
  ggplot(aes(avg)) + 
  geom_histogram(bins = 30, color = "black") + 
  xlab("Average movie rating") +
  ylab("Movie count")

#calculating the movie effect by removing mu from the rating
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i=mean(rating-mu))

#looking at the b_i estimates
movie_avgs %>% 
  qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#creating new model and testing
#predicted rating values for the model
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs,by="movieId") %>%
  .$b_i

#calculating RMSE for movie effect model
model_1_rmse <- RMSE(validation$rating,predicted_ratings)

#adding result to RMSE results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     rmse=model_1_rmse))
rmse_results %>% knitr::kable()
#-------------------------------------------------------------------------------------------

#Add the user affect on the ratings to the model--------------------------------------------
# for example, some users could be generous, and other's cranky
# Model: y_u_i=mu+b_i+b_u+e_u_i

#looking at the mean rating by user
edx %>% group_by(userId) %>% 
  summarize(avg=mean(rating)) %>%
  ggplot(aes(avg)) + 
  geom_histogram(bins = 30, color = "black") + 
  xlab("Average movie rating") +
  ylab("User count")

#calculating prediction for user effect
user_avgs <- edx %>%
  left_join(movie_avgs,by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=mean(rating-mu-b_i))

#calculating predictions including the movie and user effect 
predicted_ratings <- validation %>%
  left_join(movie_avgs,by="movieId") %>%
  left_join(user_avgs,by="userId") %>%
  mutate(pred=mu+b_i+b_u) %>%
  .$pred

#calculating the RMSE of the prediction model
model_2_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",
                                     rmse=model_2_rmse))
rmse_results %>% knitr::kable()

#looking at a potential genre and effect--------------------------------------------------------

#calculate b_g for model
genres_avgs <- edx %>% 
  group_by(genres) %>%
  summarize(b_g=mean(rating-mu))


#recalculate user effect, b_u for model
user_avgs_genre <- edx %>%
  left_join(genres_avgs,by="genres") %>%

  group_by(userId) %>%
  summarize(b_u=mean(rating-mu-b_g))

#calculating predictions including the genre, movie and user effect 
predicted_ratings <- validation %>%
  left_join(genres_avgs, by="genres") %>%
  left_join(user_avgs_genre,by="userId") %>%
  mutate(pred=mu+b_g+b_u) %>%
  .$pred

#calculating the RMSE of the prediction model
model_genre_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genre + User Effects Model",
                                     rmse=model_genre_rmse))
rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------------------

#looking closer at the movie effect---------------------------------------------------------


#when only using the movie factor
# the movies that we were the farthest off on the prediction
validation %>%
  left_join(movie_avgs,by="movieId") %>%
  mutate(residual=rating-(mu+b_i)) %>%
  arrange(desc(abs(residual))) %>%
  select(title,residual) %>%
  slice(1:10) %>%
  knitr::kable()

#top 10 best and worst movies and movie effect
movie_titles <- edx %>%
  select(movieId,title) %>%
  distinct()
# best
movie_avgs %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title,b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# worst
movie_avgs %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(b_i) %>%
  select(title,b_i) %>%
  slice(1:10) %>%
  knitr::kable()

#how often rated, best
edx %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  knitr::kable()

#how often rated, worst
edx %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  knitr::kable()

#looking to shrink the movie effect------------

#shrinking movie effect if small # of ratings
lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i=sum(rating-mu)/(n()+lambda),n_i=n())

#now top 10 are
edx %>%
  count(movieId) %>%
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title,b_i,n) %>%
  slice(1:10) %>%
  knitr::kable()

#now worst 10 are
edx %>%
  count(movieId) %>%
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles,by="movieId") %>%
  arrange(b_i) %>%
  select(title,b_i,n) %>%
  slice(1:10) %>%
  knitr::kable()

#new model with regularized movie effect
predicted_ratings <- validation %>%
  left_join(movie_reg_avgs,by="movieId") %>%
  mutate(pred=mu+b_i) %>%
  .$pred

model_3_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie Effect",
                                     rmse=model_3_rmse))
rmse_results %>% knitr::kable()

#looking for better lambda for movie effect as well as adding back the user effect as well
# choosing lambda for above with cross-validation
lambdas <- seq(0,10,0.25)

mu <- mean(edx$rating)
just_the_sum <- edx %>%
  group_by(movieId) %>%
  summarize(s=sum(rating-mu),n_i=n())

rmses <- sapply(lambdas,function(l){
  predicted_ratings <- validation %>%
    left_join(just_the_sum,by="movieId") %>%
    mutate(b_i=s/(n_i+l)) %>%
    mutate(pred=mu+b_i) %>%
    .$pred
  return(RMSE(validation$rating,predicted_ratings))
})

#plot of lambda values vs. RMSE
plot(lambdas,rmses)

#lambda with best RMSE for the movie effect
lambdas[which.min(rmses)]

#best RMSE for the movie effect
rmses[lambdas[which.min(rmses)]]

#now, include user effect as well, 
# so that if users have a small number of ratings we shrink their value as well
rmses <- sapply(lambdas,function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i=sum(rating-mu)/(n()+l))
  
  b_u <- edx %>%
    left_join(b_i,by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating-b_i-mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>%
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    mutate(pred=mu+b_i+b_u) %>%
    .$pred
  return(RMSE(validation$rating,predicted_ratings))
})

#plot of lambda values vs. RMSE
plot(lambdas,rmses)

#optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

# new calcs using above for User + Movie effect
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i=sum(rating-mu)/(n()+lambda))

b_u <- edx %>%
  left_join(b_i,by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()+lambda))

predicted_ratings <- 
  validation %>%
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  mutate(pred=mu+b_i+b_u) %>%
  .$pred

model_4_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effects Model",
                                     rmse=model_4_rmse))
rmse_results %>% knitr::kable()

#end of model development---------------------------------------------------------------------------------
