# Evaluating the Relationship Between CEO Headshots and Firm Success
By Tom Widdowson, Maya Zucker, Kai Berezniak, and Jesse Coulter

## Research Question
For our final project, we want to use AI to better understand the relationship between firm performance and facial attributes of their CEO. If we find strong correlation between any of the attributes we are evaluating and firm performance, this may suggest a way to predict future firm performance based on who a firms CEO is. More broadly, this type of technology could be used to predict an individuals performance or success simply based off their appearance. This may completely change the way individuals are hired in the future and adds a layer to the discussion around ethical uses for AI. 

### Research Question
What attributes of a CEO’s image best predict a firm's performance?

#### Our Hypothesis
We predict the attributes with the greatest relational statistical significance to firm performance will be race, sex, and emotion.


## Necessary Data

### Our Final Dataset

|Year|Firm|CEO|Sex|Race|Age|Emotion.angry|Emotion.disgust|Emotion.fear|Emotion.happy|Emotion.sad|Emotion.surprise|Emotion.neutral|Attractiveness|Firm Return|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|2010|APPL|Steve|M|W|55|.55|.45|0|0|0|0|0|.7|.1|
|2011|APPL|Tim|M|W|51|.6|.4|0|0|0|0|0|.5|.12|

### What is the observation?
The observation is firm-year.

### What is the sample period?
Our sample period is 10 years, 2010 - 2019.

### What are the sample conditions?
To be included in our dataset, we used the sample conditions of being a publicly listed firm in the S&P 500, falling within the technology sector, having financial data available from 2010 - 2019, and having a publicly accessible headshot of the firms’ CEO(s) during the 2010 - 2019 time period.

### What variables are absolutely necessary and what would you like to have if possible?
The variables that are absolutely necessary are CEO name, stock return, sex, race, age, emotion - anger, emotion - disgust, emotion - fear, emotion - happy, emotion - sad, emotion - surprise, emotion - neutral. 

The variables that we'd like to have have are additional emotion score characteristics (disgust, fear, anger, joy, etc…), attractiveness, P/E ratio of each firm, Big 5 personality traits, difference between firm and return and overall industry performance.

### What data do we have and what data do we need?
The data that we currently have is the list of companies in the S&P 500 which we can segment out to be only those in the technology sector. 

The data that we need to collect is the list of CEOs in tech companies from years 2010 - 2019, CEO Images (ideally one unique headshot for each year), and firm returns.

### How will we collect more data?
To collect more data we will use Wikipedia, Google images, and Yahoo finance.

### What are the raw inputs and how will you store them (the folder structure(s) for each input type)?
Our raw inputs are S&P 500 firms (.csv), Images (zip file with .jpeg), CEOs (.csv), Firm returns (pull directly into df from YF). 

### Speculate at a high level (not specific code!) about how you’ll transform the raw data into the final form.
Psuedocode:
1. Gather list of S&P500 firms
2. Filter on tech firms
3. Create a dataframe with the observations being firm and year
4. For each each year and firm, pull corresponding CEO
5. Add CEOs to df
6. For each year and CEO, pull headshot and store in a zip file
7. For each picture, open and run packages to identify scores
8. Save Scores to dataframe
9. Repeat this process until every picture is scored
10. For each year and CEO, pull firm returns from Yahoo finance and store in df




