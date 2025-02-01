from jiwer import wer

# Reference transcript (ground truth)
reference = """
So, anyone who's been paying attention for the last few months has been seeing headlines
like this, especially in education.
The thesis has been students are going to be using chat GPT and other forms of AI to cheat,
do their assignments, they're not going to learn, and it's going to completely undermine
education as we know it.
Now, what I'm going to argue today is not only are there ways to mitigate all of that,
if we put the right guardrails, we do the right things, we can mitigate it,
but I think we're at the cost of using AI for probably the biggest positive transformation
that education has ever seen.
And the way we're going to do that is by giving every student on the planet
an artificially intelligent but amazing personal tutor,
and we're going to give every teacher on the planet
an amazing artificially intelligent teaching assistant.
And just to appreciate how big of a deal it would be to give everyone a personal tutor,
I show you this clip from Benjamin Bloom's 1984 Two Sigma study,
or he called it the Two Sigma problem.
The Two Sigma comes from two standard deviations, Sigma the symbol for standard deviation,
and he had good data that showed that, look, a normal distribution,
that's the one that you see in the traditional bell curve right in the middle,
that's how the world sorts itself out,
that if you were to give personal one-to-one tutoring for students,
then you could actually get a distribution that looks like that right.
It says tutorial one-to-one with the asterisks,
like that right distribution, a two standard deviation improvement.
Just to put that in plain language, that could take your average student
and turn them into an exceptional student,
it can take your below average student
and turn them into an above average student.
Now, the reason why he framed it as a problem was he said,
well, this is all good, but how do you actually scale group instruction this way?
How do you actually give it to everyone in an economic way?
What I'm about to show you is I think the first moves towards doing that.
Obviously, we've been trying to approximate it in some way at Khan Academy
for over a decade now,
but I think we're at the cusp of accelerating it dramatically.
I'm going to show you the early stages of what RAI,
which we call KhanMigo, what it can now do
and maybe a little bit of where it is actually going.
So this right over here is a traditional exercise
that you or many of your children might have seen on Khan Academy,
but what's new is that little bot thing at the right,
and we'll start by seeing one of the very important safeguards,
which is the conversation is recorded and viewable by your teacher.
It's moderated actually by a second AI,
and also it does not tell you the answer.
"""

# Hypothesis transcript (ASR output)
hypothesis = """
So, anyone who's been paying attention for the last few months has been seeing headlines
like this, especially in education.
The thesis has been students are going to be using chat GPT and other forms of AI to cheat,
do their assignments, they're not going to learn, and it's going to completely undermine
education as we know it.
Now, what I'm going to argue today is not only are there ways to mitigate all of that,
if we put the right guardrails, we do the right things, we can mitigate it,
but I think we're at the cost of using AI for probably the biggest positive transformation
that education has ever seen.
And the way we're going to do that is by giving every student on the planet
an artificially intelligent but amazing personal tutor,
and we're going to give every teacher on the planet
an amazing artificially intelligent teaching assistant.
And just to appreciate how big of a deal it would be to give everyone a personal tutor,
I show you this clip from Benjamin Bloom's 1984 Two Sigma study,
or he called it the Two Sigma problem.
The Two Sigma comes from two standard deviations, Sigma the symbol for standard deviation,
and he had good data that showed that, look, a normal distribution,
that's the one that you see in the traditional bell curve right in the middle,
that's how the world sorts itself out,
that if you were to give personal one-to-one tutoring for students,
then you could actually get a distribution that looks like that right.
It says tutorial one-to-one with the asterisks,
like that right distribution, a two standard deviation improvement.
Just to put that in plain language, that could take your average student
and turn them into an exceptional student,
it can take your below average student
and turn them into an above average student.
Now, the reason why he framed it as a problem was he said,
well, this is all good, but how do you actually scale group instruction this way?
How do you actually give it to everyone in an economic way?
What I'm about to show you is I think the first moves towards doing that.
Obviously, we've been trying to approximate it in some way at Khan Academy
for over a decade now,
but I think we're at the cusp of accelerating it dramatically.
I'm going to show you the early stages of what RAI,
which we call KhanMigo, what it can now do
and maybe a little bit of where it is actually going.
So this right over here is a traditional exercise
that you or many of your children might have seen on Khan Academy,
but what's new is that little bot thing at the right,
and we'll start by seeing one of the very important safeguards,
which is the conversation is recorded and viewable by your teacher.
It's moderated actually by a second AI,
and also it does not tell you the answer.
"""

# Calculate WER
error_rate = wer(reference, hypothesis)
print(f"Word Error Rate (WER): {error_rate * 100:.2f}%")
