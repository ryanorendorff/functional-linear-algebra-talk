# This package can be called with the nixpkg function
# `agdaPackages.callPackage`, which is where the `standard-library` input comes
# from.

{ lib, mkDerivation, standard-library, functional-linear-algebra }:

mkDerivation {
  version = "1.0";
  pname = "FormalizingLinearAlgebraAlgorithms";

  buildInputs = [ standard-library functional-linear-algebra ];

  src = lib.sourceFilesBySuffices ./. [
    ".agda"
    ".lagda"
    ".lagda.md"
    ".lagda.rst"
    ".lagda.tex"
    ".agda-lib"
  ];

  meta = with lib; {
    homepage = "https://github.com/ryanorendorff/formal-linear-algebra-talk";
    description = ''
      Formalizing Linear Algebra Algorithms using Dependently Typed Functional
      Programming
    '';
    license = licenses.bsd3;
    platforms = platforms.unix;
    maintainers = with maintainers; [ ryanorendorff ];
  };
}
