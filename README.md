# axelrod-ML
Rather than query dozens of game theorists for strategies for an iterated prisoner's dilemna like Axelrod did, what if we build simple neural net models to play strategies in an iterated prisoner's dilemna? We can give them the power to evolve over generations and then see how the population changes over time and what strategies emerge.

## What's here?
We've implemented a iterated prisoner's dilemna and a number of strategies to deploy. Many strategies follow simple rules, but we've also implemented a simple recurrent neural network (default 8 free parametres) that is capable at least of evolving a large number of named strategies. Strategies can compete with each other in a tournament and the losing strategies can be removed and the successful ones copied to evolve the environment, over time. Additionally, strategies can be copied with mutations to allow for changes within the strategy space and not just in the overall environment.

## Getting started
We're still getting started too. Currently, running game.py will let you play your own strategy against the famous "tit-for-tat". The UI for tournament is still under development, though currently it runs a tournament with a set of actors whose strategy is dictated by the recurrent neural net. In the example tournament, the collaborating strategies are removed and the environment briefly enters a period where virtually all actors choose to betray, but two tit-for-tat like strategies are able to find each other and out compete the remaining agents!

## TODO
Visualization methods for showing the actor and strategies. Perhaps using Blender??
