hello,

here we have a lotto 1d cnn-dense net, designed for checking whether lotto (polish game like powerball) is really stochastic.

made for fun and for knowledge, and for high energy bills.

the network has 49 parts - as there are 6 numbers drawn out of [1,...,49]. Kind of bayesian approach.

net is untested - previous one optimized to stable numbers, which didnt change (earned with them 3, then 3, then 4 and 3, then another 4). maybe this one will be better.
ah... previous numbers:

  13.703472
  
  20.929033
  
  28.303934
  
  35.617138
  
  43.11871

Kind of golden center...

another_gambler.py is written for extra pensja game, which has two sets of numbers. it represents old architecture, without dynamic branches.

aha - dynamic version of another_gambler run just fine on gpu (laptop) with 4 GB RAM. Nice
