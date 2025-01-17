//  imports
const express = require("express");

// middleware

// check for routing purposes trial number
const checkStageOfExperiment = (req, res, next) => {
  const currentTrialNumber = req.session.trialNumber;
  switch (currentTrialNumber) {
    case 1:
      req.session.getScales = !req.session.getScales;

      if (req.session.getScales) {
        return res.redirect("/scales/getScale");
      } else {
        return next();
      }
    case 2:
      req.session.condition = "noAdvisor";
      return next();
    case 3:
      req.session.getScales = !req.session.getScales;
      if (req.session.getScales) {
        return res.redirect("/scales/getScale");
      } else {
        return res.redirect("/participant/feedback");
      }
    default:
      return next();
  }
};

const router = express.Router();

// Information view as home
router.get("/", (req, res) => {
  res.render("information.ejs");
});

// Consent routes
router.get("/consent", (req, res) => {
  res.render("consent.ejs");
});

router.post("/consent", (req, res) => {
  res.redirect("/login");
});

// description of experiment
router.get("/description", (req, res) => {
  res.render("description.ejs");
});

router.get("/webgazerconfig", (req, res) => {
  res.render("webgazerConfig.ejs");
});

// instructions view
router.get("/instructions", (req, res) => {
  const condition = req.session.condition;
  res.render("instructions.ejs", { condition: condition });
});

// rules view
router.get("/rules", checkStageOfExperiment, (req, res) => {
  res.render("rules.ejs");
});

// debrief routes
router.get("/debrief", (req, res) => {
  res.render("debrief.ejs");
});
router.get("/thanks", (req, res) => {
  res.render("thanks.ejs");
});

router.post("/debrief", (req, res) => {
  res.render("thanks.ejs");
});

module.exports = router;
