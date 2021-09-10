let

  # The functional-linear-algebra library is now an agda package in nixpkgs!
  # :-D
  pkgs = import ./nixpkgs.nix;

in pkgs.agdaPackages.callPackage ./default.nix { }
