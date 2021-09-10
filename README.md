Formalizing Linear Algebra Algorithms using Dependently Typed Functional Programming
====================================================================================

[![Presentation Type Checks](https://github.com/ryanorendorff/functional-linear-algebra-talk/workflows/Presentation%20Type%20Checks/badge.svg)][agda-literate-mode]
![Presentation PDF Builds](https://github.com/ryanorendorff/functional-linear-algebra-talk/workflows/Presentation%20PDF%20Builds/badge.svg)

[![Download Presentation PDF](https://img.shields.io/badge/-Download%20Presentation%20PDF-blue)](https://github.com/ryanorendorff/functional-linear-algebra-talk/raw/gh-pages/FormalizingLinearAlgebraAlgorithms.pdf)
[![Download Handout PDF](https://img.shields.io/badge/-Download%20Handout%20PDF-blue)](https://github.com/ryanorendorff/functional-linear-algebra-talk/raw/gh-pages/FormalizingLinearAlgebraAlgorithms-handout.pdf)


This is a presentation about the Agda library
[functional-linear-algebra][FLA], which formalizes the matrix-free
functional representation of linear algebra.


Abstract
--------

Linear algebra is the backbone of many critical algorithms such as self driving
cars and machine learning. Modern tooling makes it easy to program with linear
algebra, but the resulting code is prone to bugs from index mismatches and
improperly defined matrices.

In this talk, we will formalize basic linear algebra operations by representing
a matrix as a function from one vector space to another. This "matrix-free"
construction will enable us to prove basic properties about linear algebra; from
this base, we will show a framework for formulating optimization problems that
is correct by construction, meaning that it will be impossible to represent
improperly formed matrices. We will compare the Agda framework to similar
frameworks written in Python and in dependently typed Haskell, and demonstrate
proving properties about optimization algorithms using this framework.


How to play around with the Agda code in this repository
--------------------------------------------------------

If you have the [Nix][nix] package manager installed, you can run

```
nix-shell
```

at the root of this presentation and then launch emacs 

```
emacs src/FunctionalPresentation.lagda.md
```

More information on the Agda emacs mode can be found
[here][agda-emacs-mode]. If you use [Spacemacs][spacemacs], the
documentation for its Agda mode is [here][spacemacs-agda-mode].


How to build the presentation
-----------------------------

```
nix-build presentation.nix
```

<!-- References -->

[FLA]: https://github.com/ryanorendorff/functional-linear-algebra
[agda-literate-mode]: https://agda.readthedocs.io/en/v2.6.1.1/tools/literate-programming.html
[nix]: https://nixos.org
[agda-emacs-mode]: https://agda.readthedocs.io/en/v2.6.1.1/tools/emacs-mode.html
[spacemacs]: https://www.spacemacs.org/
[spacemacs-agda-mode]: https://www.spacemacs.org/layers/+lang/agda/README.html
