<img src="/Pics/Screen Shot 2016-08-05 at 1.59.56 PM.png" align = 'middle'> 

##Overview##

 

Stylish is a two week data science project made by Scott Contri for his Galvanize capstone project that aims to identify the style of a writer based off his/her similarity to a famous author using natural language processing (NLP). It currently uses a Random Forest model to make its predictions based off a training set consisting of 46 authors  from my personal collection of ebooks (in .txt format). The model runs on an associated website named Stylish, designed and programmed by Karen Kelly, using an EC2 instance from Amazon Web Services (AWS). 

 <img src="/Pics/Screen Shot 2016-08-05 at 3.43.43 PM.png" align = 'middle'> 
 <img src="/Pics/Screen Shot 2016-08-05 at 3.44.30 PM.png" align = 'middle'> 
 <img src="/Pics/Screen Shot 2016-08-05 at 1.58.52 PM.png" align = 'middle'>

 
##Motivation##

  

*"The limits of my language mean the limits of my world."*

*-Ludwig Wittgenstein*

  

Everyday people tell themselves stories in their heads. These stories are told in voices unique to them and with a particular cadence depending on the terrain there mind is wandering. Using a personalized style to create and express their dreams, ideas, and of course, stories, people try to present themselves using various mediums. Often people try to express themselves through writing. Everyone from Shakespeare (NLP was used to investigate suspicions of the playwright’s collaboration with others) to the unabomber, Ted Kaczynski (whose brother turned him in after recognizing Ted’s writing style in the unabomber’s manifesto), have a unique linguistic style. However, although almost anyone can clearly read the difference in style between Hamlet and Industrial Society and Its Future, differentiating between two writers is usually a challenge for most people. 

In literature, style is loosely defined as the way an author uses words. With such a vacant definition it is obviously difficult to measure a writer’s style, which is the challenge stylometry ventures to take on. Stylometry applies the study of linguistic style and is currently used to identify authorship of documents. Unfortunately, no method has yet been produced to accurately identify different styles amongst a large amount of documents. The number of applications that could be granted by a program that could perform this incredible task are virtually limitless. Recommendations for books could be provided to readers based off of the style of books they have previously enjoyed reading. The efficiency of publication could be enhanced by syncing the style publishers were selling with the style authors were offering. Perhaps one day machines could even produce interpretable stories that would fascinate and terrify us, but in the meantime I’d really just like to get a better understanding of my own writing style. Maybe you would too.

  

##Data##

  

The data for this project came from my personal collection of ebook novels that I converted to txt format. The data was cleaned so that only the body of the novels were left (e.g. table of contents, appendix, etc. were removed). A total of 46 authors and 181 novels were used to train the model.

 <img src="/Pics/Screen Shot 2016-08-05 at 3.45.53 PM.png" align = 'middle'> 
  

##Model##

  

Feature engineering was performed on the data to derive more signal from its content. The data was split into documents consisting of 5000 words/tokens each. The features engineered out of the data were inspired by stylometry research performed by Congzhou Ramyaa and Khaled Rasheed He, the features included are:

1. type-token ratio: A ratio comparing the types of words used to the total amount of words. The higher the ratio, the more varied the vocabulary. It also reflects an author’s tendency to repeat words.  
2. mean word length: Longer words are traditionally associated with more pedantic and formal styles, whereas shorter words are a typical feature of informal spoken language.  
3. mean sentence length: Longer sentences are often the indicator of carefully planned writing, while shorter sentences are more characteristic of spoken language. 
4. number of commas: Commas signal the ongoing flow of ideas within a sentence.  
5. number of semicolons: Semicolons indicate the reluctance of an author to stop a sentence where (s)he could.  
6. number of quotation marks: Frequent use of quotations is considered a typical involvement-feature.  
7. number of exclamation marks: Exclamations tend to signal strong emotions. 
8. number of question marks: Questions can signal a reflective style.  
9. number of ands: Ands are markers of coordination, which, as opposed to subordination, is more frequent in spoken production.  
10. number of buts: The contrastive linking buts are markers of coordination too.  
11. number of howevers: The conjunct “however” is meant to form a contrastive pair with “but”.  
12. number of ifs: If-clauses are samples of subordination.  
13. number of thats: Most of the thats are used for subordination while a few are used as demonstratives.  
14. number of mores: ‘More’ is an indicator of an author’s preference for comparative structure.  
15. number of musts: Modal verbs are potential candidates for expressing tentativeness. Musts are more often used non-epistemically.  
16. number of mights: Mights are more often used epistemically. 
17. number of thiss: Thiss are typically used for anaphoric reference.  
18. number of verys: Verys are stylistically significant for its emphasis on its modifiees 
  

Additionally, the text was transformed using tf-idf vectorization so that the computer could process it. Stop words, names, and locations were removed and the text was also stemmed (i.e. words were cut down to their roots). Principal component analysis was performed on the vectorized text in order to reduce the dimensionality. The top ten dimensions (the dimensions in which the most variance occurred), were extracted for each document and added to the feature engineered dataframe. 


##Results##


Random forest classification was performed on the the dataframe and was fitted onto a model. The cross-validation accuracy was 83%, meaning 83% of the predictions on the test data (extracted from 20% of the total original data) were correct. New data was run on this model in order to be classified as a given author. 

<img src="/Pics/Screen Shot 2016-08-05 at 3.46.33 PM.png" align = 'middle'> 

##Web App##

  
An API was produced from the model and was used to form the backend of the web app called *Stylish*, which takes writing from the user and returns the top 5 authors the user is most similar to along with the normalized percentage of similarity.

<img src="/Pics/Screen Shot 2016-08-05 at 2.02.01 PM.png" align = 'middle'>
